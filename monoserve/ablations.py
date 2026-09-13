"""Ablations of MonoServe (the ablation study of the evaluation).

  /ML    SingleLaneEngine here: no multi-lane execution. One batch at a time
         on every SM; each iteration carries every decoding request's next
         token and a chunk of the waiting prompts, as in a continuous-
         batching engine with chunked prefill. The control plane still
         prices each batch and picks its kernel form and link credits.
  /MCA   EngineConfig(contention_aware=False): the plan search without
         gamma and the fabric without the credit gate.
  /KF    GreenEngine here: no kernel fabric. The lanes run on green
         contexts, one SM partition each, with nothing borrowed; shares
         change only through a re-carve that stalls every lane, and link
         credits become grid sizes.
  forms  EngineConfig(forms=(64,)) or (0,): one kernel form for DRAM experts.
  no staging  EngineConfig(staging_experts=0).

The engine here has the interface of monoserve.engine.Engine, so the vLLM
integration and the tests drive it the same way.
"""
import itertools
import math
import time

import numpy as np

from monoserve.control import (Estimator, EstimatorOptions, LaneWork, Seq, load_calibration,
                               model_shape, profile_view)
from monoserve.engine import DECODE, NO_LANE, Engine, EngineConfig, PagePool, _Req
from monoserve.fabric import Fabric
from monoserve.placement import ActivationProfile
from monoserve.runtime.kinds import PAGE
from monoserve.runtime.kv import KVCache
from monoserve.runtime.lane import Lane
from monoserve.runtime.requests import RequestTable

LOOKAHEAD_PAGES = 2


class SingleLaneEngine:
    """One lane on every SM, one mixed batch per iteration (the /ML ablation)."""

    def __init__(self, cfg, load_weights, calibration, config=None, decode_profile=None,
                 prefill_profile=None, check_stops=True, log=print):
        self.cfg, self.log, self.check_stops = cfg, log, check_stops
        conf = self.conf = config or EngineConfig()
        self.fab = Fabric(pool_bytes=conf.pool_bytes)
        self.S = self.fab.num_workers
        L, E = cfg.num_layers, cfg.num_experts
        self.weights = load_weights(self.fab, staging_slots=0, staging_buffers=1)
        self.kv = KVCache(self.fab, cfg, conf.kv_pages)
        self.kv.free.remove(0)
        self.pool = PagePool(self.kv)
        self.table = RequestTable(self.fab, conf.max_slots, -(-conf.max_len // PAGE))
        self.lane = Lane(self.fab, cfg, self.weights, self.kv, self.table, 0, "prefill",
                         max_tokens=conf.token_budget,
                         max_reqs=conf.max_decode + conf.max_batch_requests,
                         max_len=conf.max_len, seed=conf.seed)
        cal = calibration if hasattr(calibration, "sms") else load_calibration(calibration)
        if cal.sms != self.S:
            raise ValueError(f"the calibration is for {cal.sms} SMs, this GPU has {self.S}")
        opts = EstimatorOptions()
        opts.contention_aware = conf.contention_aware
        self.est = Estimator(model_shape(cfg), cal, opts)
        prefill_p = (np.full((L, E), 1.0 / E) if prefill_profile is None
                     else np.asarray(prefill_profile, dtype=np.float64))
        self.view = profile_view((decode_profile or ActivationProfile(L, E)).p, prefill_p,
                                 self.weights.hot)
        self.reqs, self.by_rid = {}, {}
        self.keys = itertools.count(1)
        self.free_slots = list(range(conf.max_slots))[::-1]
        self.queue = []        # requests with prompt tokens left, oldest first
        self.decoding = []     # requests past their prompt
        self.prefilled = {}    # key -> prompt tokens computed
        self.last = {}         # key -> last output token
        self.running = None    # (handle, gen, entries) of the batch on the GPU
        self.t0 = time.monotonic()

    def now(self):
        return time.monotonic() - self.t0

    def start(self):
        self.fab.start()

    def stop(self):
        self.fab.stop()

    def idle(self):
        return not self.reqs and self.running is None

    # ------------------------------------------------------------------
    def add_request(self, rid, prompt, max_tokens, temperature=0.0, eos=(), ttft=None, tpot=None,
                    arrival=None):
        prompt = list(prompt)
        if len(prompt) + 1 > self.conf.max_len:
            return [(rid, [], "rejected")]
        key = next(self.keys)
        reserve = min(max_tokens, self.conf.reserve_tokens)
        total = min(self.conf.max_len, -(-(len(prompt) + reserve) // PAGE) * PAGE)
        self.reqs[key] = _Req(rid, key, prompt, max_tokens, total - len(prompt), temperature,
                              frozenset(eos), ttft or 0.0, tpot or 0.0)
        self.by_rid[rid] = key
        self.prefilled[key] = 0
        self.queue.append(key)
        if self.running is None:
            self.launch()
        return []

    def finish(self, rid):
        key = self.by_rid.pop(rid, None)
        if key is None:
            return []
        r = self.reqs[key]
        r.done = True
        if key in self.queue:
            self.queue.remove(key)
        if key in self.decoding:
            self.decoding.remove(key)
        if self.running is None or all(key != e[0] for e in self.running[2]):
            self.free(key)
        return []

    def free(self, key):
        r = self.reqs.pop(key)
        self.prefilled.pop(key, None)
        self.last.pop(key, None)
        if r.slot >= 0:
            self.pool.release(r.rid, r.pages)
            self.free_slots.append(r.slot)

    # ------------------------------------------------------------------
    def pick_form(self, seqs, pairs):
        """The kernel form (and its link slots) the estimator prices lowest
        for this batch on every SM."""
        w = LaneWork()
        w.passes = [seqs]
        w.deadline = 1e9
        for layer in range(self.cfg.num_layers):
            a, m = self.view.expected(False, layer, pairs)
            w.active.append(a)
            w.miss.append(m)
        load = self.est.load(w)
        q = self.est.link_credits
        form = min(self.conf.forms, key=lambda f: self.est.evaluate(load, self.S, q, f).total)
        cap = max(1, int(q // self.est.calibration.q_form.get(form, 1.0)))
        return form, cap if self.conf.contention_aware else 1 << 20

    def launch(self):
        """Build and publish the next batch: every decoding request's next
        token, then prompt chunks, oldest first, up to the token budget."""
        budget = self.conf.token_budget
        entries = []
        for key in self.decoding:
            r = self.reqs[key]
            entries.append((key, [self.last[key]], len(r.prompt) + r.delivered - 1, True))
            budget -= 1
        for key in self.queue:
            if budget <= 0 or len(entries) >= self.lane.R:
                break
            r = self.reqs[key]
            if r.slot < 0:
                if not self.free_slots:
                    break
                try:
                    r.pages = self.pool.grow(r.rid, [], -(-(len(r.prompt) + r.reserve) // PAGE))
                except MemoryError:
                    break
                r.slot = self.free_slots.pop()
                self.table.set_pages(r.slot, r.pages)
                self.table.set_temperature(r.slot, r.temperature)
                self.table.reset_steps(r.slot)
            p0 = self.prefilled[key]
            n = min(budget, len(r.prompt) - p0)
            entries.append((key, r.prompt[p0:p0 + n], p0, p0 + n == len(r.prompt)))
            budget -= n
        if not entries:
            return
        reqs = [dict(slot=self.reqs[k].slot, tokens=toks, pos0=p0, pages=self.reqs[k].pages,
                     last=last) for k, toks, p0, last in entries]
        h = self.lane.build_prefill(reqs)
        form, cap = self.pick_form([Seq(len(t), p0) for _, t, p0, _ in entries],
                                   sum(len(t) for _, t, _, _ in entries) * self.cfg.top_k)
        self.fab.publish(map=[0] * self.S, caps=[cap], order=[0], programs=[h], forms=[form])
        self.running = (h, self.fab.generation(h), entries)

    def step(self, wait=0.0):
        until = time.monotonic() + wait
        while True:
            out = self.poll()
            if out or time.monotonic() >= until:
                return out
            time.sleep(20e-6)

    def poll(self):
        if self.running is None:
            if self.queue or self.decoding:
                self.launch()
            return []
        h, gen, entries = self.running
        m = self.fab.progress()["lanes"][0]
        if m["gen"] != gen or m["running"]:
            return []
        self.running = None
        self.lane.release(h)
        out = []
        for key, toks, p0, last in entries:
            r = self.reqs.get(key)
            if r is None:
                continue
            if r.done:
                self.free(key)
                continue
            if key in self.queue:   # a prompt chunk
                self.prefilled[key] = p0 + len(toks)
                if not last:
                    continue
                self.queue.remove(key)
                self.decoding.append(key)
            n = self.table.produced(r.slot)
            new = self.table.tokens(r.slot, r.delivered, n)
            r.delivered = n
            if new:
                self.last[key] = new[-1]
            reason = None
            if self.check_stops:
                for i, t in enumerate(new):
                    if t in r.eos:
                        new, reason = new[:i + 1], "stop"
                        break
                if reason is None and n >= r.max_tokens:
                    new, reason = new[:len(new) - (n - r.max_tokens)], "length"
            if reason is None and not self.ensure_pages(r, len(r.prompt) + n):
                reason = "length"
            out.append((r.rid, new, reason))
            if reason is not None:
                self.by_rid.pop(r.rid, None)
                self.decoding.remove(key)
                self.free(key)
        self.launch()
        return out

    def ensure_pages(self, r, tokens):
        need = min(-(-tokens // PAGE) + LOOKAHEAD_PAGES, -(-self.conf.max_len // PAGE))
        if need <= len(r.pages):
            return tokens < self.conf.max_len
        try:
            r.pages = self.pool.grow(r.rid, r.pages, need)
        except MemoryError:
            return tokens + PAGE <= len(r.pages) * PAGE
        self.table.set_pages(r.slot, r.pages)
        return True


GRANULE = 8   # SMs: the device splits into green-context groups of this size


class GreenEngine(Engine):
    """The lanes on green contexts instead of the kernel fabric (the /KF
    ablation). Each lane's workers are one launch of the fabric kernel in
    the lane's own SM partition, and nothing is borrowed, so a lane runs on
    its partition alone. A plan's shares take effect only through a
    re-carve: every launch ends at its current tile, the partitions are
    carved anew, and the lanes resume in the new ones; all lanes stall
    meanwhile. Link credits become grid sizes: a lane's link slots are its
    credits in whole tiles, rounded up, set when its program starts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from monoserve import _C
        self.parts = _C.GreenPartitions()
        self.layout = ()     # the shares of the current carve, in granules per lane
        self.ranges = {}     # lane -> (first worker, workers)
        self.grid = {}       # lane -> (program, link slots) set when the program started
        self.stats.update(recarves=0, recarve_ms=[])

    def start(self):
        self.carve([DECODE], {DECODE: self.S})
        if self.host_loop:
            self.host_loop.start()

    def stop(self):
        if self.host_loop:
            self.host_loop.stop()
        self.fab.stop()
        self.parts.destroy()
        self.layout, self.ranges = (), {}
        self.log_decisions()

    def carve(self, order, want):
        """End every launch, carve one partition per lane in `order` (the
        last lane takes the SMs the others leave), and launch each lane's
        workers in its own partition."""
        t0 = time.perf_counter()
        self.fab.stop()
        self.parts.destroy()
        # the first split is in whole granules; later ones split a remainder
        ask = [max(GRANULE, want[l] // GRANULE * GRANULE) for l in order[:-1]]
        while ask and sum(ask) + GRANULE > self.S:
            ask[ask.index(max(ask))] -= GRANULE
        ask.append(self.S - sum(ask) if ask else self.S // GRANULE * GRANULE)
        for _ in range(64):
            try:
                granted = self.parts.create(ask)
                break
            except RuntimeError:     # a split the driver cannot grant
                self.parts.destroy()
                if ask[-1] - 2 >= GRANULE:
                    ask[-1] -= 2
                elif len(ask) > 1 and max(ask[:-1]) > GRANULE:
                    i = ask.index(max(ask[:-1]))
                    ask[i] -= GRANULE
                    ask[-1] = self.S - sum(ask[:-1])
                else:
                    raise
        else:
            raise RuntimeError("green partitions: no carve fits the plan")
        self.ranges, first = {}, 0
        for lane, k in zip(order, granted):
            self.ranges[lane] = (first, k)
            first += k
        for i, lane in enumerate(order):
            self.fab.start_on(self.parts.stream(i), self.parts.context(i), self.ranges[lane][1],
                              self.ranges[lane][0])
        self.grid = {}
        self.stats["recarves"] += 1
        self.stats["recarve_ms"].append((time.perf_counter() - t0) * 1e3)

    def publish(self):
        want, slack, form, credits = {}, {}, {}, {}
        for i, lane in enumerate(self.plan_lanes):
            lp = self.plan.lanes[i]
            want[lane], slack[lane], form[lane], credits[lane] = lp.sms, lp.slack, lp.form, lp.credits
        # a lane still finishing a program the plan no longer covers keeps a
        # partition until the program ends
        for lane in range(self.n_lanes):
            if lane not in want and (lane in self.passes or (lane == DECODE and (
                    self.dec_handle is not None or self.dec_pending is not None))):
                want[lane], slack[lane] = GRANULE, math.inf
        layout = tuple(sorted((lane, want[lane] // GRANULE) for lane in want))
        if want and layout != self.layout:
            self.carve(sorted(want, key=lambda lane: -slack[lane]), want)
            self.layout = layout
        n = self.n_lanes
        mapping, caps, forms = [NO_LANE] * self.S, [0] * n, [64] * n
        for lane, (first, k) in self.ranges.items():
            mapping[first:first + k] = [lane] * k
            caps[lane] = 1           # a lane finishing its program keeps one slot
        for lane in form:
            forms[lane] = form[lane]
            h = self.programs[lane]
            if self.grid.get(lane, (None,))[0] != h:   # a new program: its grid size
                q = self.est.calibration.q_form.get(form[lane], 1.0)
                cap = math.ceil(credits[lane] / q) if self.conf.contention_aware else 1 << 20
                self.grid[lane] = (h, max(1, cap))
            caps[lane] = self.grid[lane][1]
        self.fab.publish(map=mapping, caps=caps, order=[NO_LANE], programs=list(self.programs),
                         forms=forms)
        self.stats["published"] += 1
