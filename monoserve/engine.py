"""The MonoServe engine: requests in, tokens out.

The engine owns the fabric and everything its lanes use: the model weights
with the hot tier and one staging buffer per prefill lane, the KV cache,
the request table, one decode lane (two buffer sets, so a new decode
program can be built while the current one runs) and up to two prefill
lanes, the host loop that fills the staging buffers, and the control plane.
Admission events (an arrival, a finished prefill pass, a finished request,
a lane behind the pace its plan published) go to the C++ admission; each
decision becomes lane programs and one published plan: SM shares as the
worker-to-lane map, link slots, and kernel forms.

step() runs one round: it detects finished passes and decode-program
switches, checks lanes against their pace, and returns the tokens the
device produced since the previous round. The caller decides when a
request stops (finish()), or lets the engine check EOS and max_tokens.
"""
import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from monoserve.control import (Admission, AdmissionConfig, Estimator, EstimatorOptions,
                               LaneWork, PlanSearch, Request, SearchOptions, Seq,
                               load_calibration, model_shape, profile_view)
from monoserve.fabric import KIND, Fabric, Program
from monoserve.fabric.program import tile_array
from monoserve.placement import ActivationProfile
from monoserve.runtime.host_loop import HostLoop
from monoserve.runtime.kinds import PAGE
from monoserve.runtime.kv import KVCache
from monoserve.runtime.lane import Lane
from monoserve.runtime.requests import RequestTable

NO_LANE = 255
DECODE = 0            # fabric lane of the decode batch; prefill lanes follow
LOOKAHEAD_PAGES = 2   # KV pages kept ahead of a decoding request


@dataclass
class EngineConfig:
    max_slots: int = 512            # requests holding a slot (admitted, not yet freed)
    kv_pages: int = 4096            # KV-cache pages of 64 tokens
    max_len: int = 32768            # longest request, prompt and output
    token_budget: int = 16384       # prompt tokens per prefill pass
    max_batch_requests: int = 64    # requests per prefill batch
    max_decode: int = 256           # decode batch capacity
    prefill_lanes: int = 2
    staging_experts: int = 0        # V: experts per layer each staging buffer half holds
    reserve_tokens: int = 1024      # output tokens reserved up front; more pages come on demand
    alpha: float = 5.0              # SLO scale for requests that bring no targets
    forms: tuple = (16, 32, 64, 128, 0)
    contention_aware: bool = True   # False: plan without gamma, run without the credit gate
    seed: int = 0
    behind_margin: float = 1.25     # a lane this much slower than its estimate is behind
    profile_interval: float = 1.0   # seconds between decode-profile updates
    pool_bytes: int = 1 << 30       # the fabric's program pool


@dataclass
class _Req:
    rid: object
    key: int
    prompt: list
    max_tokens: int
    reserve: int                    # output tokens reserved at admission
    temperature: float
    eos: frozenset
    ttft: float
    tpot: float
    slot: int = -1
    pages: list = field(default_factory=list)
    delivered: int = 0
    done: bool = False


class PagePool:
    """KV pages for requests, from the engine's own cache. vLLM's scheduler
    substitutes a pool backed by its KV-cache manager (monoserve.vllm)."""

    def __init__(self, kv):
        self.kv = kv

    def grow(self, rid, pages, n):
        """The request's pages, at least n of them; MemoryError when the
        cache has too few left."""
        if n <= len(pages):
            return pages
        return pages + self.kv.allocate(n - len(pages))

    def release(self, rid, pages):
        """The device no longer uses the request's pages."""
        self.kv.release(pages)


class Engine:
    def __init__(self, cfg, load_weights, calibration, config=None, decode_profile=None,
                 prefill_profile=None, check_stops=True, log=print):
        """load_weights(fab, staging_slots=..., staging_buffers=...) returns
        the ModelWeights (its hot tier chosen by the caller). calibration:
        a Calibration, a dict, or a calibration file. check_stops: the
        engine ends requests on EOS and max_tokens itself (a caller such as
        the vLLM scheduler that checks stops passes False)."""
        self.cfg, self.log, self.check_stops = cfg, log, check_stops
        conf = self.conf = config or EngineConfig()
        self.fab = Fabric(pool_bytes=conf.pool_bytes)
        self.S = self.fab.num_workers
        L, E = cfg.num_layers, cfg.num_experts
        self.weights = load_weights(self.fab, staging_slots=2 * conf.staging_experts,
                                    staging_buffers=conf.prefill_lanes)
        self.kv = KVCache(self.fab, cfg, conf.kv_pages)
        self.kv.free.remove(0)   # page 0 absorbs writes past a request's pages
        self.pool = PagePool(self.kv)
        self.table = RequestTable(self.fab, conf.max_slots, -(-conf.max_len // PAGE))
        self.decode_lanes = [Lane(self.fab, cfg, self.weights, self.kv, self.table, DECODE, "decode",
                                  max_reqs=conf.max_decode, max_len=conf.max_len, seed=conf.seed)
                             for _ in range(2)]
        self.prefill_lanes = [Lane(self.fab, cfg, self.weights, self.kv, self.table, 1 + i, "prefill",
                                   max_tokens=conf.token_budget, max_reqs=conf.max_batch_requests,
                                   max_len=conf.max_len, seed=conf.seed + 1 + i)
                              for i in range(conf.prefill_lanes)]
        self.prefill_p = (np.full((L, E), 1.0 / E) if prefill_profile is None
                          else np.asarray(prefill_profile, dtype=np.float64))
        self.host_loop = None
        if conf.staging_experts > 0:
            self.host_loop = HostLoop(self.fab, self.weights)
            for i, lane in enumerate(self.prefill_lanes):
                self.host_loop.add_prefill_lane(lane, i, profile=self.prefill_p)

        # The control plane.
        cal = calibration if hasattr(calibration, "sms") else load_calibration(calibration)
        if cal.sms != self.S:
            raise ValueError(f"the calibration is for {cal.sms} SMs, this GPU has {self.S}")
        opts = EstimatorOptions()
        opts.contention_aware = conf.contention_aware
        self.model = model_shape(cfg)
        self.est = Estimator(self.model, cal, opts)
        so = SearchOptions()
        so.forms = list(conf.forms)
        self.search = PlanSearch(self.est, so)
        self.decode_profile = decode_profile or ActivationProfile(L, E)
        self.view = profile_view(self.decode_profile.p, self.prefill_p, self.weights.hot)
        ac = AdmissionConfig()
        ac.max_prefill_lanes = conf.prefill_lanes
        ac.token_budget = conf.token_budget
        ac.max_batch_requests = conf.max_batch_requests
        ac.max_decode = conf.max_decode
        ac.staging_experts = conf.staging_experts
        ac.kv_bytes_per_token = self.model.kv_bytes_per_token
        ac.kv_capacity = (conf.kv_pages - 1) * PAGE * self.model.kv_bytes_per_token
        self.adm = Admission(self.est, self.search, self.view, ac)

        p = Program()
        p.stage(tile_array(KIND["nop"]))
        self.nop = p.upload(self.fab, first=0, iterations=1)
        self.n_lanes = 1 + conf.prefill_lanes
        self.programs = [0] * self.n_lanes            # program each lane should run
        self.plan_lanes, self.plan = [], None
        self.reqs, self.by_rid = {}, {}
        self.keys = itertools.count(1)
        self.free_slots = list(range(conf.max_slots))[::-1]
        self.active = set()                           # requests with a slot, not finished
        self.release_after = {}                       # lane -> requests to free when it lets go
        # decode lane: running program and batch, and a switch in flight
        self.dec_handle, self.dec_buf, self.dec_batch = None, 0, []
        self.dec_pending = None                       # (handle, gen, buffer, batch)
        self.dec_target = []
        self.dec_iters = (0, None)                    # (iterations, host time) at the last check
        self.dec_slow = 0
        # prefill passes: lane -> dict(handle, gen, start, members, mark_ns, flagged)
        self.passes = {}
        self.hist_seen = [None, None]
        self.t0 = time.monotonic()
        self.last_profile = 0.0
        self.stats = {"searches": 0, "decisions": 0, "decision_us": [], "published": 0}

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def now(self):
        return time.monotonic() - self.t0

    def idle(self):
        """No request in flight and nothing left to free."""
        return (not self.reqs and not self.passes and self.dec_pending is None
                and not any(self.release_after.values()))

    def start(self):
        self.fab.start()
        if self.host_loop:
            self.host_loop.start()

    def stop(self):
        if self.host_loop:
            self.host_loop.stop()
        self.fab.stop()
        self.log_decisions()

    def log_decisions(self):
        """The admission decisions' latency (plan searches included)."""
        d = self.stats["decision_us"]
        if d:
            p50, p99 = np.percentile(d, [50, 99])
            self.log(f"admission decisions: {len(d)}, median {p50:.0f} us, P99 {p99:.0f} us")

    # ------------------------------------------------------------------
    # requests
    # ------------------------------------------------------------------
    def solo_targets(self, n_prompt, context=None):
        """alpha times the estimated solo latency of a request: its prompt
        alone on the whole GPU (prompts under 1K tokens take the 1K prompt's
        target), and one decode step alone at its context."""
        budget = self.conf.token_budget
        n = max(n_prompt, 1024)
        w = LaneWork()
        chunks = -(-n // budget)
        w.passes = [[Seq(min(budget, n - i * budget), i * budget)] for i in range(chunks)]
        w.staging = self.conf.staging_experts
        w.deadline = 1e9
        for layer in range(self.cfg.num_layers):
            a, m = self.view.expected(False, layer, min(n, budget) * self.cfg.top_k)
            w.active.append(a)
            w.miss.append(m)
        ttft = min(self.est.evaluate(self.est.load(w), self.S, self.est.link_credits, f).total
                   for f in self.conf.forms)
        d = LaneWork()
        d.decode = True
        d.passes = [[Seq(1, context or n_prompt)]]
        d.deadline = 1e9
        for layer in range(self.cfg.num_layers):
            a, m = self.view.expected(True, layer, self.cfg.top_k)
            d.active.append(a)
            d.miss.append(m)
        tpot = min(self.est.evaluate(self.est.load(d), self.S, self.est.link_credits, f).total
                   for f in self.conf.forms)
        return self.conf.alpha * ttft, self.conf.alpha * tpot

    def add_request(self, rid, prompt, max_tokens, temperature=0.0, eos=(), ttft=None, tpot=None,
                    arrival=None):
        """Admit a request (token ids). ttft and tpot are its targets in
        seconds; without them the engine sets alpha times its solo latency.
        Returns the rejections the decision made."""
        now = self.now()
        prompt = list(prompt)
        if len(prompt) + 1 > self.conf.max_len:
            return [(rid, [], "rejected")]
        if ttft is None or tpot is None:
            t, p = self.solo_targets(len(prompt))
            ttft, tpot = ttft or t, tpot or p
        key = next(self.keys)
        # reserve whole pages: the prompt and up to reserve_tokens of output
        reserve = min(max_tokens, self.conf.reserve_tokens)
        total = min(self.conf.max_len, -(-(len(prompt) + reserve) // PAGE) * PAGE)
        r = _Req(rid, key, prompt, max_tokens, total - len(prompt), temperature, frozenset(eos),
                 ttft, tpot)
        self.reqs[key] = r
        self.by_rid[rid] = key
        d = self.adm.arrive(Request(key, len(prompt), r.reserve, now if arrival is None else arrival,
                                    ttft, tpot), now)
        return self.apply(d, now)

    def finish(self, rid):
        """The request is done (a stop condition or an abort). Returns the
        rejections of the admission event this is: queued requests no plan
        can serve any longer, which the caller must report as finished."""
        key = self.by_rid.pop(rid, None)
        if key is None:
            return []
        r = self.reqs[key]
        r.done = True
        self.active.discard(key)
        now = self.now()
        out = self.apply(self.adm.finished(key, now), now)
        self.free_when_idle(key)
        return out

    def free_when_idle(self, key):
        """Free a finished request's slot and pages once no program uses them."""
        users = [lane for lane, p in self.passes.items() if key in p["members"]]
        if key in self.dec_batch or (self.dec_pending and key in self.dec_pending[3]):
            users.append(DECODE)
        if users:
            for lane in users:
                self.release_after.setdefault(lane, set()).add(key)
            return
        r = self.reqs.pop(key)
        if r.slot >= 0:
            self.pool.release(r.rid, r.pages)
            self.free_slots.append(r.slot)

    def assign(self, r):
        r.pages = self.pool.grow(r.rid, [], -(-(len(r.prompt) + r.reserve) // PAGE))
        r.slot = self.free_slots.pop()
        self.table.set_pages(r.slot, r.pages)
        self.table.set_temperature(r.slot, r.temperature)
        self.table.reset_steps(r.slot)
        self.active.add(r.key)

    # ------------------------------------------------------------------
    # decisions
    # ------------------------------------------------------------------
    def apply(self, d, now):
        """Carry out an admission decision: start the prefill passes it
        opened, move the decode batch, publish its plan."""
        self.stats["searches"] += d.searches
        self.stats["decisions"] += 1
        self.stats["decision_us"].append(d.micros)
        out = []
        for key in d.rejected:
            r = self.reqs.pop(key)
            self.by_rid.pop(r.rid, None)
            out.append((r.rid, [], "rejected"))
        changed = False
        for lane, spec in zip(d.pass_lanes, d.passes):
            self.start_pass(lane, spec, now)
            changed = True
        if d.publish:
            self.plan_lanes, self.plan = list(d.lanes), d.plan
            self.dec_target = list(d.decode)
            changed = True
        if self.rebuild_decode():
            changed = True
        if changed:
            self.publish()
        return out

    def start_pass(self, lane, spec, now):
        members, reqs = [], []
        for key, n, pos0, last in zip(spec.reqs, spec.tokens, spec.pos0, spec.last):
            r = self.reqs[key]
            if r.slot < 0:
                self.assign(r)
            members.append(key)
            reqs.append(dict(slot=r.slot, tokens=r.prompt[pos0:pos0 + n], pos0=pos0, pages=r.pages,
                             last=bool(last)))
        h = self.prefill_lanes[lane - 1].build_prefill(reqs)
        m = self.fab.progress()["lanes"][lane]
        self.passes[lane] = dict(handle=h, gen=self.fab.generation(h), start=now, members=set(members),
                                 mark_ns=m["mark_ns"], flagged=False)
        self.programs[lane] = h

    def rebuild_decode(self):
        """Build the decode program of the target batch on the idle buffer
        set, unless a switch is still in flight. Returns whether a new
        program was built."""
        if self.dec_pending is not None or self.dec_target == self.dec_batch:
            return False
        target = list(self.dec_target)
        if not target:
            self.dec_pending = (self.nop, self.fab.generation(self.nop), None, [])
            self.programs[DECODE] = self.nop
            return True
        buf = self.dec_buf ^ 1 if self.dec_handle is not None else self.dec_buf
        lane = self.decode_lanes[buf]
        for key in target:
            if key not in self.dec_batch:   # joins: decode starts after the prompt
                self.table.set_length(self.reqs[key].slot, len(self.reqs[key].prompt))
        slots = [self.reqs[k].slot for k in target]
        h = lane.build_decode(slots, {self.reqs[k].slot: self.reqs[k].pages for k in target},
                              iterations=0)
        self.dec_pending = (h, self.fab.generation(h), buf, target)
        self.programs[DECODE] = h
        return True

    def publish(self):
        """Publish the current plan and the lanes' programs as one epoch."""
        n = self.n_lanes
        mapping = [NO_LANE] * self.S
        caps, forms, slack = [0] * n, [64] * n, {}
        w = 0
        lanes, plan = self.plan_lanes, self.plan
        for i, lane in enumerate(lanes):
            lp = plan.lanes[i]
            k = min(lp.sms, self.S - w)
            mapping[w:w + k] = [lane] * k
            w += k
            caps[lane] = lp.cap if self.conf.contention_aware else 1 << 20
            forms[lane] = lp.form
            slack[lane] = lp.slack
        if lanes and w < self.S:
            tight = min(lanes, key=lambda l: slack[l])
            mapping[w:] = [tight] * (self.S - w)
        # borrowing: tightest lanes first, and any lane still finishing a
        # program the plan no longer covers (so it can reach its boundary)
        order = sorted(lanes, key=lambda l: slack[l])
        busy = [l for l in range(n) if l not in order and (l in self.passes or (
            l == DECODE and (self.dec_handle is not None or self.dec_pending is not None)))]
        for l in busy:
            if caps[l] == 0:
                caps[l] = 1
        self.fab.publish(map=mapping, caps=caps, order=(order + busy) or [NO_LANE],
                         programs=list(self.programs), forms=forms)
        self.stats["published"] += 1

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------
    def step(self, wait=0.0):
        """One round of the loop; waits up to `wait` seconds for tokens.
        Returns [(rid, new tokens, finish reason or None)]."""
        until = time.monotonic() + wait
        while True:
            out = self.poll()
            if out or time.monotonic() >= until:
                return out
            time.sleep(20e-6)

    def poll(self):
        now = self.now()
        out = []
        lanes = self.fab.progress()["lanes"]
        # prefill passes that finished, and their progress
        for lane, p in list(self.passes.items()):
            m = lanes[lane]
            if m["gen"] == p["gen"] and not m["running"]:
                del self.passes[lane]
                self.prefill_lanes[lane - 1].release(p["handle"])
                self.programs[lane] = 0
                for key in self.release_after.pop(lane, ()):
                    self.free_when_idle(key)
                elapsed = now - p["start"]
                est = self.lane_estimate(lane)
                if est:
                    self.est.observe(False, False, elapsed / est)
                    self.est.observe(False, True, elapsed / est)
                out += self.apply(self.adm.pass_done(lane, now), now)
                continue
            if m["mark_ns"] != p["mark_ns"]:
                layer, phase = m["mark"] >> 8, m["mark"] & 0xFF
                self.adm.progress(lane, layer + (phase == 3))
            est = self.lane_estimate(lane)
            if est and not p["flagged"] and now - p["start"] > self.conf.behind_margin * est:
                p["flagged"] = True
                out += self.apply(self.adm.behind(lane, now), now)
        # the decode lane: a switch that landed, and its pace
        m = lanes[DECODE]
        if self.dec_pending is not None and m["gen"] == self.dec_pending[1]:
            h, _, buf, batch = self.dec_pending
            if self.dec_handle is not None:
                self.decode_lanes[self.dec_buf].release(self.dec_handle)
            self.dec_handle = None if h == self.nop else h
            if buf is not None:
                self.dec_buf = buf
            self.dec_batch, self.dec_pending = batch, None
            for key in self.release_after.pop(DECODE, ()):
                self.free_when_idle(key)
            self.dec_iters = (m["iterations"], now)
            if self.rebuild_decode():
                self.publish()
        elif self.dec_handle is not None:
            out += self.check_decode_pace(m["iterations"], now)
        out += self.collect()
        if now - self.last_profile > self.conf.profile_interval:
            self.update_profile()
            self.last_profile = now
        return out

    def lane_estimate(self, lane):
        if self.plan is None or lane not in self.plan_lanes:
            return None
        return self.plan.lanes[self.plan_lanes.index(lane)].est.total

    def check_decode_pace(self, iterations, now):
        it0, t0 = self.dec_iters
        if t0 is None or iterations - it0 < 8:
            return []
        step = (now - t0) / (iterations - it0)
        self.dec_iters = (iterations, now)
        est = self.lane_estimate(DECODE)
        if not est:
            return []
        self.est.observe(True, False, step / est)
        self.est.observe(True, True, step / est)
        self.dec_slow = self.dec_slow + 1 if step > self.conf.behind_margin * est else 0
        if self.dec_slow >= 3:
            self.dec_slow = 0
            return self.apply(self.adm.behind(DECODE, now), now)
        return []

    def collect(self):
        """New tokens of every active request, from the pinned token rings."""
        out = []
        for key in list(self.active):
            r = self.reqs[key]
            n = self.table.produced(r.slot)
            if n <= r.delivered:
                continue
            toks = self.table.tokens(r.slot, r.delivered, n)
            r.delivered = n
            self.adm.generated(key, n)
            reason = None
            if self.check_stops:
                for i, t in enumerate(toks):
                    if t in r.eos:
                        toks, reason = toks[:i + 1], "stop"
                        break
                if reason is None and n >= r.max_tokens:
                    toks, reason = toks[:len(toks) - (n - r.max_tokens)], "length"
            if reason is None and not self.ensure_pages(r, len(r.prompt) + n):
                reason = "length"
            out.append((r.rid, toks, reason))
            if reason is not None:
                out += self.finish(r.rid)
        return out

    def ensure_pages(self, r, tokens):
        """Keep KV pages ahead of a decoding request. False when the cache
        has none left, or the request reached max_len."""
        need = min(-(-tokens // PAGE) + LOOKAHEAD_PAGES, -(-self.conf.max_len // PAGE))
        if need <= len(r.pages):
            return tokens < self.conf.max_len
        try:
            r.pages = self.pool.grow(r.rid, r.pages, need)
        except MemoryError:
            return tokens + PAGE <= len(r.pages) * PAGE
        self.table.set_pages(r.slot, r.pages)
        row = np.zeros((1, self.table.bt_stride), dtype=np.int32)
        row[0, :len(r.pages)] = r.pages
        programs = [(self.dec_buf, self.dec_batch)]
        if self.dec_pending is not None and self.dec_pending[2] is not None:
            programs.append((self.dec_pending[2], self.dec_pending[3]))
        for buf, batch in programs:
            if r.key in batch:
                lane = self.decode_lanes[buf]
                lane._write(lane.bt, row, rows_offset=batch.index(r.key))
        return True

    def update_profile(self):
        """Fold the decode lane's expert activations into the decode profile
        (prefill routing is not counted) and refresh the admission's view."""
        for i, lane in enumerate(self.decode_lanes):
            counts = np.frombuffer(self.fab.read(lane.hist.data_ptr(), lane.hist.numel() * 4),
                                   dtype=np.int32).reshape(lane.hist.shape).astype(np.int64)
            prev = self.hist_seen[i]
            self.hist_seen[i] = counts
            if prev is None:
                continue
            delta = counts - prev
            for layer in range(delta.shape[0]):
                self.decode_profile.update(layer, delta[layer])
        self.view.set_profile(True, self.decode_profile.p.ravel().tolist())
