"""The calibration runs (see the package docstring)."""
import dataclasses
import math
import time

import numpy as np
import torch

from monoserve import _C
from monoserve.control import Estimator, LaneWork, Seq, load_calibration, model_shape
from monoserve.fabric import KIND, Fabric, Program
from monoserve.fabric.args import pack
from monoserve.fabric.program import tile_array
from monoserve.runtime import kinds as K
from monoserve.runtime.kinds import FORM_ASYM, HEAD_DIM, PAGE, TILE_HEIGHTS
from monoserve.runtime.kv import KVCache, kv_map
from monoserve.runtime.lane import Lane, act_map, gemm_tiles
from monoserve.runtime.requests import RequestTable
from monoserve.runtime.weights import ModelWeights, weight_map

NO_LANE = 255
SECTOR = 32                # bytes of one MTQ entry
FORMS = (16, 32, 64, 128, FORM_ASYM)
REFERENCE_FORM = 64


def form_key(form):
    return "asym" if form == FORM_ASYM else str(form)


def cycle_chain(n_nodes, seed, stride=16):
    """A pointer chain visiting n_nodes nodes, 128 bytes apart, in random
    order as one cycle: slot i * stride holds the slot of the next node."""
    order = np.random.default_rng(seed).permutation(n_nodes)
    nxt = np.empty(n_nodes, dtype=np.int64)
    nxt[order[:-1]] = order[1:]
    nxt[order[-1]] = order[0]
    buf = np.zeros(n_nodes * stride, dtype=np.int64)
    buf[np.arange(n_nodes) * stride] = nxt * stride
    return buf


def k_splits(kt, per=K.ASYM_MAX_KT):
    """The asymmetric kernel's K ranges, cut as the expander cuts them."""
    n = -(-kt // per)
    step = -(-kt // n)
    return [(k0, min(step, kt - k0)) for k0 in range(0, kt, step)]


def asym_block(rows):
    return 16 if rows <= 16 else 32 if rows <= 32 else 64


def monotone(values):
    """Measured rates made non-decreasing in the share or the credits."""
    return np.maximum.accumulate(np.asarray(values, dtype=np.float64)).tolist()


class Calibrator:
    """Allocates every buffer the runs need, then (in run()) starts the
    fabric, measures, and returns the calibration as a dict in the format
    of monoserve.control.calibration."""

    def __init__(self, cfg, quick=False, experts=None, pool_bytes=1 << 30, log=print):
        self.cfg, self.quick, self.log = cfg, quick, log
        self.model = model_shape(cfg)
        self.fab = fab = Fabric(pool_bytes=pool_bytes)
        self.S = fab.num_workers
        H, I = cfg.hidden, cfg.intermediate
        hq, hkv, D = cfg.num_heads, cfg.num_kv_heads, HEAD_DIM
        self.E = min(cfg.num_experts, experts or cfg.num_experts)
        self.w = 3 * H * I * 2   # bytes of one expert
        bf, f32 = torch.bfloat16, torch.float32
        gen = torch.Generator(device="cuda").manual_seed(0)

        def rnd(*shape):
            return (torch.randn(*shape, device="cuda", generator=gen) * 0.02).to(bf)

        def zeros(*shape, dtype=bf):
            return torch.zeros(*shape, dtype=dtype, device="cuda")

        # Experts, in HBM and in pinned CPU DRAM.
        self.w13, self.w2 = rnd(self.E, 2 * I, H), rnd(self.E, H, I)
        self.w13_host, self.w2_host = self.w13.cpu().pin_memory(), self.w2.cpu().pin_memory()
        self.maps = {"hbm": (weight_map(fab, self.w13), weight_map(fab, self.w2)),
                     "host": (weight_map(fab, self.w13_host, pinned=True),
                              weight_map(fab, self.w2_host, pinned=True))}
        self.compute_experts = min(4, self.E)
        self.compute_rows = 256 if quick else 1024
        rows = max(self.E * 128, self.compute_experts * self.compute_rows)
        self.xp, self.h = rnd(rows, H), zeros(rows, I)
        self.acc, self.ws = zeros(rows, H, dtype=f32), zeros(rows, 2 * I, dtype=f32)
        self.pair_tok = torch.arange(rows, dtype=torch.int32, device="cuda")
        self.pair_w = torch.full((rows,), 1e-3, dtype=f32, device="cuda")
        self.xp_maps = {h: act_map(fab, self.xp, h) for h in TILE_HEIGHTS}
        self.h_maps = {h: act_map(fab, self.h, h) for h in TILE_HEIGHTS}

        # Dense GEMM: a wide weight for a compute shape and a read shape.
        self.dense_n = 4096 if quick else 32768
        self.dense_m = 512 if quick else 2048
        self.wd, self.xd = rnd(self.dense_n, H), rnd(self.dense_m, H)
        self.yd = zeros(self.dense_m, self.dense_n)
        self.md = weight_map(fab, self.wd)
        self.xd_maps = {h: act_map(fab, self.xd, h) for h in TILE_HEIGHTS}

        # Attention: one long prefill, and a decode batch over long contexts.
        self.P = 1024 if quick else 8192
        self.B, self.ctx = (8, 1024) if quick else (64, 4096)
        self.chunk_blocks = 16
        n_pages = self.P // PAGE + self.B * (self.ctx // PAGE)
        self.kc, self.vc = rnd(n_pages, PAGE, hkv, D), rnd(n_pages, PAGE, hkv, D)
        self.k_map, self.v_map = kv_map(fab, self.kc), kv_map(fab, self.vc)
        self.qp, self.op = rnd(self.P, hq, D), zeros(self.P, hq, D)
        self.wsp = (zeros(self.P, hq, D, dtype=f32), zeros(self.P, hq, dtype=f32),
                    zeros(self.P, hq, dtype=f32))
        self.bt_p = torch.arange(self.P // PAGE, dtype=torch.int32, device="cuda")[None]
        self.info_p = torch.tensor([[0, 0, self.P, self.P]], dtype=torch.int32, device="cuda")
        self.qp_map = fab.tensor_map(self.qp.data_ptr(), [64, self.P, 2, hq, 1],
                                     [hq * D * 2, 128, D * 2, self.P * hq * D * 2], [64, 64, 2, 1, 1])
        self.max_chunks = -(-self.ctx // (self.chunk_blocks * PAGE))
        self.qd, self.od = rnd(self.B, hq, D), zeros(self.B, hq, D)
        self.wsd = (zeros(self.B, hq, self.max_chunks, D, dtype=f32),
                    zeros(self.B, hq, self.max_chunks, dtype=f32))
        per = self.ctx // PAGE
        self.bt_d = (self.P // PAGE + torch.arange(self.B * per, dtype=torch.int32,
                                                   device="cuda")).view(self.B, per)
        self.info_d = torch.tensor([[r, self.ctx - 1, 1, self.ctx] for r in range(self.B)],
                                   dtype=torch.int32, device="cuda")
        self.qd_map = fab.tensor_map(self.qd.data_ptr(), [64, self.B * hq, 2, 1, 1],
                                     [D * 2, 128, self.B * hq * D * 2, self.B * hq * D * 2],
                                     [64, 64, 2, 1, 1])

        # Probe chains far larger than the L2, in HBM and in pinned CPU DRAM.
        mb = (64, 32) if quick else (512, 256)
        self.chain_hbm = torch.from_numpy(cycle_chain((mb[0] << 20) >> 7, 1)).cuda()
        self.chain_host = torch.from_numpy(cycle_chain((mb[1] << 20) >> 7, 2)).pin_memory()
        self.chain_host_dev = _C.host_device_pointer(self.chain_host.data_ptr())
        self.probe_out = zeros(8, dtype=torch.int64)
        self.loads = 500 if quick else 4000

        grid = [8, 16, 24, 32, 48, 64, 96, 128]
        self.shares = sorted({s for s in grid if s < self.S} | {self.S})
        if quick:
            self.shares = sorted({min(8, self.S), self.S // 2, self.S})
        self.nops = {}   # lane -> its no-op program (a program runs on one lane at a time)
        self.R_H = self.R_C = None

        # Real layer programs for the per-layer and per-pass overheads: decode
        # steps of one request through one and two MoE layers of the model's
        # shape (synthetic weights, every expert in HBM). Built here, since
        # nothing may allocate once the fabric runs.
        self.oh_ctx = 100
        d = getattr(cfg, "dense_layers", 0)
        self.oh_layers = (d + 1, d + 2)
        cfg_n = dataclasses.replace(cfg, num_layers=self.oh_layers[1])
        self.oh_weights = ModelWeights.synthetic(
            fab, cfg_n, [list(range(cfg.num_experts)) for _ in range(self.oh_layers[1])])
        per = -(-(self.oh_ctx + 64) // PAGE)
        self.oh_kv = KVCache(fab, cfg_n, per)
        self.oh_req = RequestTable(fab, 1, per)
        self.oh_pages = self.oh_kv.allocate(per)
        self.oh_lanes = [Lane(fab, dataclasses.replace(cfg, num_layers=n), self.oh_weights, self.oh_kv,
                              self.oh_req, 0, "decode", max_reqs=1) for n in self.oh_layers]

    # ------------------------------------------------------------------
    # programs
    # ------------------------------------------------------------------
    def blob(self, name, **fields):
        data = pack(name, **fields)
        addr = self.fab.blob_alloc(len(data))
        self.fab.write(addr, data)
        return addr

    def upload(self, stages, iterations=0):
        """stages: [(tiles, link)], each after the one before; the program
        repeats until replaced unless iterations says otherwise."""
        p = Program()
        prev = None
        for tiles, link in stages:
            s = p.stage(tiles, link=link)
            if prev is not None:
                p.after(prev, s)
            prev = s
        return p.upload(self.fab, first=0, iterations=iterations)

    def dense_program(self, rows, bm, n):
        H = self.cfg.hidden
        args = self.blob("GemmArgs", out=self.yd, ld_out=self.dense_n, k_tiles=H // 64,
                         epi=K.EPI_STORE, bm=bm, n_limit=self.dense_n)
        return self.upload([(gemm_tiles(args, self.md, self.xd_maps[bm], n, rows, bm), False)])

    def expert_program(self, n_experts, rows, form, where="hbm"):
        """The MoE window over n_experts experts of `rows` rows each: w13,
        the reduce under the asymmetric form, and w2. Weights in CPU DRAM
        make the expert stages link stages."""
        H, I = self.cfg.hidden, self.cfg.intermediate
        m13, m2 = self.maps[where]
        link = where == "host"
        e = np.arange(n_experts)
        if form == FORM_ASYM:
            bm = asym_block(rows)
            a13 = self.blob("GemmArgs", out=self.ws, ld_out=2 * I, k_tiles=H // 64,
                            epi=K.EPI_ATOMIC_F32, bm=bm, n_limit=I, up_offset=I)
            a2 = self.blob("GemmArgs", out=self.acc, ld_out=H, k_tiles=I // 64, epi=K.EPI_WADD,
                           bm=bm, n_limit=H, pair_token=self.pair_tok, pair_weight=self.pair_w)
            red = self.blob("ReduceArgs", ws=self.ws, out=self.h, I=I, ld_ws=2 * I, ld_out=I)

            def split_tiles(args, wm, xm, features, step, kt):
                sp = k_splits(kt)
                ee, ff, ss = (a.ravel() for a in np.meshgrid(e, np.arange(0, features, step),
                                                             np.arange(len(sp)), indexing="ij"))
                k0 = np.array([a for a, _ in sp])[ss]
                kn = np.array([b for _, b in sp])[ss]
                return tile_array(K.GEMM_ASYM, i0=ff, i1=ee * rows, i2=rows, i3=ee, a0=args,
                                  a1=wm, a2=xm, a3=k0 | (kn << 16))

            er, rr = (a.ravel() for a in np.meshgrid(e, np.arange(0, rows, 64), indexing="ij"))
            t_red = tile_array(K.ASYM_REDUCE, i0=er * rows + rr, i1=np.minimum(64, rows - rr),
                               a0=red)
            return self.upload([(split_tiles(a13, m13, self.xp_maps[bm], I, 64, H // 64), link),
                                (t_red, False),
                                (split_tiles(a2, m2, self.h_maps[bm], H, 128, I // 64), link)])
        bm = form
        a13 = self.blob("GemmArgs", out=self.h, ld_out=I, k_tiles=H // 64, epi=K.EPI_SILU_MUL,
                        bm=bm, n_limit=I, up_offset=I)
        a2 = self.blob("GemmArgs", out=self.acc, ld_out=H, k_tiles=I // 64, epi=K.EPI_WADD,
                       bm=bm, n_limit=H, pair_token=self.pair_tok, pair_weight=self.pair_w)
        stages = []
        for args, wm, xm, features, step in ((a13, m13, self.xp_maps[bm], I, 64),
                                             (a2, m2, self.h_maps[bm], H, 128)):
            ee, bb, ff = (a.ravel() for a in np.meshgrid(e, np.arange(0, rows, bm),
                                                         np.arange(0, features, step), indexing="ij"))
            stages.append((tile_array(K.GEMM, i0=ff, i1=ee * rows + bb, i2=np.minimum(bm, rows - bb),
                                      i3=ee, a0=args, a1=wm, a2=xm), link))
        return self.upload(stages)

    def expert_work(self, n_experts, rows, form):
        """(FLOPs, HBM bytes) of expert_program as the estimator counts them."""
        H, I = self.cfg.hidden, self.cfg.intermediate
        bm = asym_block(rows) if form == FORM_ASYM else form
        flops = 6.0 * H * I * math.ceil(rows / bm) * bm * n_experts
        if form == FORM_ASYM:
            return flops, n_experts * (self.w + rows * _C.control.asym_row_bytes(self.model))
        return flops, n_experts * self.w * math.ceil(rows / bm)

    def attn_args(self, q_map, out, ws, bt, info, stride, max_chunks):
        hq, hkv = self.cfg.num_heads, self.cfg.num_kv_heads
        return self.blob("AttnArgs", q_map=q_map, k_map=self.k_map, v_map=self.v_map, out=out,
                         ws_o=ws[0], ws_m=ws[1], ws_l=ws[2] if len(ws) > 2 else 0, block_table=bt,
                         req_info=info, scale_log2=math.log2(math.e) / math.sqrt(HEAD_DIM),
                         bt_stride=stride, hq=hq, hkv=hkv, group=hq // hkv,
                         chunk_blocks=self.chunk_blocks, max_chunks=max_chunks)

    def prefill_attn_program(self):
        hq = self.cfg.num_heads
        args = self.attn_args(self.qp_map, self.op, self.wsp, self.bt_p, self.info_p,
                              self.P // PAGE, 1)
        ck = self.chunk_blocks * PAGE
        stages = []
        for c in range(-(-self.P // ck)):
            qbs = [qb for qb in range(-(-self.P // 128)) if min(self.P, qb * 128 + 128) > c * ck]
            qq, hh = (a.ravel() for a in np.meshgrid(qbs, np.arange(hq), indexing="ij"))
            stages.append((tile_array(K.ATTN_PREFILL, i0=0, i1=hh, i2=qq, i3=c, a0=args), False))
        return self.upload(stages)

    def decode_attn_program(self):
        hkv = self.cfg.num_kv_heads
        args = self.attn_args(self.qd_map, self.od, self.wsd, self.bt_d, self.info_d,
                              self.ctx // PAGE, self.max_chunks)
        n = self.max_chunks
        rr, gg, cc = (a.ravel() for a in np.meshgrid(np.arange(self.B), np.arange(hkv),
                                                     np.arange(n), indexing="ij"))
        stages = [(tile_array(K.ATTN_DECODE, i0=rr, i1=gg, i2=0, i3=cc, a0=args), False)]
        if n > 1:
            stages.append((tile_array(K.ATTN_COMBINE, i0=np.arange(self.B), i1=n, a0=args), False))
        return self.upload(stages)

    def attn_work(self, decode):
        from monoserve.control import LaneWork, Seq
        w = LaneWork()
        w.decode = decode
        w.passes = [[Seq(1, self.ctx - 1) for _ in range(self.B)]] if decode else [[Seq(self.P, 0)]]
        return _C.control.pass_load(self.model, w)

    def probe_program(self, where):
        base = self.chain_hbm.data_ptr() if where == "hbm" else self.chain_host_dev
        return self.upload([(tile_array(KIND["probe"], i0=self.loads, i1=0, a0=base,
                                        a1=self.probe_out.data_ptr()), False)])

    # ------------------------------------------------------------------
    # running and timing
    # ------------------------------------------------------------------
    def publish(self, plan):
        """plan: [(lane, program, workers, link slots)]; workers are taken
        in order from worker 0, the rest idle, and nothing is borrowed, so
        every lane runs on exactly its share."""
        mapping = [NO_LANE] * self.S
        n_lanes = 1 + max(lane for lane, *_ in plan)
        programs, caps = [0] * n_lanes, [0] * n_lanes
        w = 0
        for lane, handle, workers, cap in plan:
            mapping[w:w + workers] = [lane] * workers
            w += workers
            programs[lane], caps[lane] = handle, cap
        if w > self.S:
            raise ValueError("calibration plan needs more SMs than the GPU has")
        self.fab.publish(map=mapping, caps=caps, order=[NO_LANE], programs=programs)

    def time_lane(self, plan, lane, count=6, warmup=2, timeout=600.0):
        """Median iteration time (s) of `lane` under the plan, from the
        fabric's in-kernel timers."""
        start = self.fab.iterations(lane)
        self.publish(plan)
        samples = self.fab.iteration_times(lane, start, warmup + count, timeout)
        if not samples or samples[-1][0] < start + warmup + count:
            raise RuntimeError(f"calibration: lane {lane} did not finish its iterations")
        d = [(t1 - t0) / (i1 - i0) for (i0, t0), (i1, t1) in zip(samples, samples[1:])
             if i0 >= start + warmup]
        return float(np.median(d)) * 1e-9

    def stop(self, plan, release=()):
        """Let the plan's lanes finish on one no-op iteration, then free
        the given programs."""
        for lane, *_ in plan:
            if lane not in self.nops:
                self.nops[lane] = self.upload([(tile_array(KIND["nop"]), False)], iterations=1)
        # the current iteration must still finish, link tiles included, so
        # the stopping lanes keep their link slots until they are idle
        self.publish([(lane, self.nops[lane], workers, 1 << 20) for lane, _, workers, _ in plan])
        t0 = time.time()
        for lane, *_ in plan:
            while self.fab.lane_stats(lane)["running"]:
                if time.time() - t0 > 600:
                    raise RuntimeError(f"calibration: lane {lane} did not stop")
                time.sleep(1e-3)
        for h in release:
            self.fab.release(h)

    def sweep(self, handle):
        """Iteration time of a program alone at every share."""
        times = [self.time_lane([(0, handle, s, 0)], 0) for s in self.shares]
        self.stop([(0, handle, self.S, 0)], release=[handle])
        return times

    def rates(self, compute, read):
        """Rates of a kernel from its compute shape (program, FLOPs) and its
        read shape (program, bytes), plus its entries in flight per SM."""
        tc, tr = self.sweep(compute[0]), self.sweep(read[0])
        gbps = [read[1] / t / 1e9 for t in tr]
        s0 = self.shares[0]
        return {"sms": self.shares,
                "tflops": monotone([compute[1] / t / 1e12 for t in tc]),
                "gbps": monotone(gbps),
                "q_per_sm": gbps[0] * 1e9 / s0 * self.R_H / SECTOR}

    def probe_latency(self, plan, reads=3):
        """Latency (s) of one miss of the probe lane (lane 1) under the plan."""
        vals = []
        for _ in range(reads):
            self.time_lane(plan, 1, count=1, warmup=1)
            ps = np.frombuffer(self.fab.read(self.probe_out.data_ptr(), 8), dtype=np.int64)[0]
            vals.append(float(ps) * 1e-12)
        return float(np.median(vals))

    def layer_overheads(self, out):
        """(per-layer, per-pass) overhead in seconds: what real layer
        programs take beyond the estimator's windows. The difference of
        decode steps through one and two MoE layers, less the estimator's
        windows for one layer, is the per-layer overhead; what is left of
        the one-layer step is the pass's."""
        req, pages, ctx = self.oh_req, self.oh_pages, self.oh_ctx
        times = []
        for lane in self.oh_lanes:
            req.set_pages(0, pages)
            req.set_temperature(0, 0.0)
            req.reset_steps(0)
            req.set_length(0, ctx)
            h = lane.build_decode([0], {0: pages})
            times.append(self.time_lane([(0, h, self.S, 0)], 0, count=8, warmup=3))
            self.stop([(0, h, self.S, 0)])
            lane.release(h)
        base = dict(out, layer_overhead_us=0.0, tail_overhead_us=0.0)
        est = []
        for n in self.oh_layers:
            e = Estimator(model_shape(dataclasses.replace(self.cfg, num_layers=n)), load_calibration(base))
            w = LaneWork()
            w.decode = True
            w.passes = [[Seq(1, ctx)]]
            w.active, w.miss = [float(self.cfg.top_k)] * n, [0.0] * n
            w.deadline = 1.0
            est.append(e.evaluate(e.load(w), self.S, e.link_credits, REFERENCE_FORM).total)
        layer = max(0.0, (times[1] - times[0]) - (est[1] - est[0]))
        tail = max(0.0, times[0] - est[0] - self.oh_layers[0] * layer)
        return layer, tail

    # ------------------------------------------------------------------
    def run_all(self):
        cfg, S, log = self.cfg, self.S, self.log
        H = cfg.hidden
        out = {"device": torch.cuda.get_device_name(), "sms": S, "reference_form": REFERENCE_FORM,
               "model": {"hidden": H, "intermediate": cfg.intermediate, "experts": cfg.num_experts,
                         "top_k": cfg.top_k, "layers": cfg.num_layers, "heads": cfg.num_heads,
                         "kv_heads": cfg.num_kv_heads}}

        # Residency of a miss, alone.
        probe_h, probe_c = self.probe_program("hbm"), self.probe_program("host")
        self.R_H = self.probe_latency([(1, probe_h, 1, 0)])
        self.R_C = self.probe_latency([(1, probe_c, 1, 0)])
        self.stop([(1, probe_c, 1, 0)], release=[probe_c])
        out["R_H_us"], out["R_C_us"] = self.R_H * 1e6, self.R_C * 1e6
        log(f"R_H {self.R_H * 1e9:.0f} ns, R_C {self.R_C * 1e9:.0f} ns")

        # Solo sweeps.
        n_c = min(8192, self.dense_n)
        out["dense"] = self.rates(
            (self.dense_program(self.dense_m, 128, n_c), 2.0 * self.dense_m * n_c * H),
            (self.dense_program(16, 16, self.dense_n), 2.0 * self.dense_n * H))
        dec, pre = self.attn_work(True), self.attn_work(False)
        out["attn_decode"] = self.rates((self.decode_attn_program(), dec["attn_flops"]),
                                        (self.decode_attn_program(), dec["kv_bytes"]))
        # Prefill attention is compute-bound in every shape; its read rate is
        # the read-bound (decode) attention tiles'.
        pre_rates = self.rates((self.prefill_attn_program(), pre["attn_flops"]),
                               (self.decode_attn_program(), dec["kv_bytes"]))
        out["attn_prefill"] = pre_rates
        out["expert"] = {}
        for form in FORMS:
            read_rows = 16 if form == FORM_ASYM else form
            out["expert"][form_key(form)] = self.rates(
                (self.expert_program(self.compute_experts, self.compute_rows, form),
                 self.expert_work(self.compute_experts, self.compute_rows, form)[0]),
                (self.expert_program(self.E, read_rows, form),
                 self.expert_work(self.E, read_rows, form)[1]))
            log(f"expert {form_key(form)}: {out['expert'][form_key(form)]['tflops'][-1]:.0f} TFLOP/s, "
                f"{out['expert'][form_key(form)]['gbps'][-1]:.0f} GB/s at {S} SMs")

        # The heaviest HBM-reading kernel: its read shape is the background
        # of the credit sweeps and the load of the gamma probe.
        kinds = {"dense": out["dense"], "attn_decode": out["attn_decode"],
                 **{f"expert{k}": v for k, v in out["expert"].items()}}
        heavy = max(kinds, key=lambda k: kinds[k]["q_per_sm"])
        out["q_max"] = kinds[heavy]["q_per_sm"]

        def background():
            if heavy == "dense":
                return self.dense_program(16, 16, self.dense_n)
            if heavy == "attn_decode":
                return self.decode_attn_program()
            f = heavy[len("expert"):]
            form = FORM_ASYM if f == "asym" else int(f)
            return self.expert_program(self.E, 16 if form == FORM_ASYM else form, form)

        # Credit sweeps, one per form.
        out["rho"], out["q_form"] = {}, {}
        lw = S // 2
        # a fine grid: the plan search reads the curve's bend below saturation
        caps = [c for c in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128) if c <= lw]
        bg = background()
        for form in FORMS:
            rows = 64 if form == FORM_ASYM else form
            h = self.expert_program(self.E, rows, form, where="host")
            link_bytes = self.E * self.w
            rates = [link_bytes / self.time_lane([(0, h, lw, c), (1, bg, S - lw, 0)], 0) for c in caps]
            self.stop([(0, h, lw, 0)], release=[h])
            q = rates[0] * self.R_C / SECTOR
            out["q_form"][form_key(form)] = q
            out["rho"][form_key(form)] = {"credits": [c * q for c in caps],
                                          "gbps": monotone([r / 1e9 for r in rates])}
            log(f"credits {form_key(form)}: q {q:.0f}, {rates[-1] / 1e9:.1f} GB/s at {caps[-1]} slots")
        self.stop([(1, bg, S - lw, 0)])
        out["beta_gbps"] = max(max(v["gbps"]) for v in out["rho"].values())

        # Gamma: the probe's miss while other lanes load the MTQ.
        q_bg, q_ref = out["q_max"], out["q_form"][form_key(REFERENCE_FORM)]
        link = self.expert_program(self.E, REFERENCE_FORM, REFERENCE_FORM, where="host")
        levels = [(0.0, 1.0)]
        for n in sorted({8, 16, 32, 64, 96, S - 1}):
            if n < S:
                lat = self.probe_latency([(1, probe_h, 1, 0), (0, bg, n, 0)])
                levels.append((n * q_bg * self.R_H, lat / self.R_H))
        self.stop([(0, bg, S - 1, 0)])
        lw = min(64, (S - 1) // 2)
        for cap in (4, 16, 64):
            if cap <= lw:
                lat = self.probe_latency([(1, probe_h, 1, 0), (2, link, lw, cap)])
                levels.append((cap * q_ref * self.R_C, lat / self.R_H))
        n = S - 1 - lw
        cap = min(64, lw)
        lat = self.probe_latency([(1, probe_h, 1, 0), (0, bg, n, 0), (2, link, lw, cap)])
        levels.append((n * q_bg * self.R_H + cap * q_ref * self.R_C, lat / self.R_H))
        self.stop([(1, probe_h, 1, 0), (0, bg, n, 0), (2, link, lw, 0)], release=[probe_h, bg, link])
        levels.sort()
        xs, ys = [], []
        for x, y in levels:   # merge equal exposures, keep the stretch non-decreasing
            y = max(1.0, y, ys[-1] if ys else 1.0)
            if xs and x <= xs[-1]:
                ys[-1] = max(ys[-1], y)
            else:
                xs.append(x)
                ys.append(y)
        out["gamma"] = {"exposure_entry_us": [x * 1e6 for x in xs], "stretch": ys}
        log("gamma: " + ", ".join(f"{x * 1e6:.0f}:{y:.2f}" for x, y in zip(xs, ys)))

        # Stage transitions (a chain of one-tile stages, for reference), and
        # the overheads of real layer programs.
        chain = self.upload([(tile_array(KIND["nop"]), False)] * 64)
        stage = self.time_lane([(0, chain, S, 0)], 0) / 64
        self.stop([(0, chain, S, 0)], release=[chain])
        out["stage_us"] = stage * 1e6
        layer, tail = self.layer_overheads(out)
        out["layer_overhead_us"], out["tail_overhead_us"] = layer * 1e6, tail * 1e6
        log(f"stage {stage * 1e6:.1f} us; overheads {layer * 1e6:.0f} us a layer, {tail * 1e6:.0f} us a pass")
        return out

    def run(self):
        """Start the fabric, measure everything, stop it; returns the dict."""
        self.fab.start()
        try:
            return self.run_all()
        finally:
            self.fab.stop()
