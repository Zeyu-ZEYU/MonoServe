"""Co-run experiments on disjoint SM partitions (Figs. 5, 6, and 15).

Two computations run at the same time on disjoint SM partitions carved by
partitions.py, and each side's co-run latency is divided by its solo
latency on the same partition. Every cell measures a triplet on one
partition geometry: side A alone, side B alone, and each side while the
other runs.

Instrument. The measured side runs eagerly (after a deep warmup) and is
timed with CUDA events around each iteration, after a GPU spin that lets
the CPU finish enqueueing. The contender is a CUDA graph holding many of
its iterations (ChainRunner), pre-enqueued so that it keeps the other
partition busy for the whole measured window; a completion-margin check
records whether it did (covered=true).

Suites:
  pairs        Fig. 5. Prefill computations at a 4K input length, each on
               its own 48% of the SMs: attention+attention, symmetric and
               asymmetric MoE kernels paired with each other and with
               attention. MoE weights sit in pinned CPU DRAM.
  reader       Fig. 6. Prefill attention (4K, 48% of the SMs) beside a
               synthetic reader that walks the expert weights of 16 layers
               at 32-byte granularity on 8 to 68 SMs, with the same bytes
               placed in CPU DRAM or in HBM.
  copy-engine  Fig. 15. The same attention beside a continuous
               host-to-device copy that walks 1 to 128 GB of pinned CPU
               DRAM through the copy engine.

    python corun.py --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --suite pairs --out results/corun_pairs.jsonl
"""
import argparse
import gc
import json
import statistics
import time

import torch
import triton
import triton.language as tl

import model_harness as mh
import moe_kernels
import partitions
from common import log, sleep_cycles
from pinned import to_pinned_cuda_view

# Mean number of distinct experts per layer that a 4K-token prefill of
# Qwen3-30B-A3B activates. MoE drivers route each token to top-k experts
# drawn from a fixed subset of this size, so they move the bytes a real
# prefill moves.
N_ACTIVE_PREFILL_4K = 104


def cuda_check(err, what):
    if int(err) != 0:
        raise RuntimeError(f"{what} failed with CUDA error {int(err)}")


# ---------------------------------------------------------------------------
# Drivers: each enqueues one unit of work on the current stream
# ---------------------------------------------------------------------------
class MoeDriver:
    """One MoE sublayer: the real router GEMM (kept for timing fidelity)
    and the expert FFNs on `tokens` rows with the chosen kernel."""

    def __init__(self, name, gate_w, w13, w2, tokens, kernel, top_k,
                 n_experts, seed, n_active):
        self.name = name
        self.kernel = kernel
        self.gate_w = gate_w
        self.w13, self.w2 = w13, w2
        self.top_k, self.E = top_k, n_experts
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.h = (torch.randn(tokens, gate_w.shape[1], generator=g)
                  * 0.5).to(torch.bfloat16).cuda()
        subset = torch.randperm(n_experts, generator=g)[:n_active]
        rows = [subset[torch.randperm(n_active, generator=g)[:top_k]]
                for _ in range(tokens)]
        self.ti = torch.stack(rows).to(torch.int32).cuda()
        tw = torch.rand(tokens, top_k, generator=g).float()
        self.tw = (tw / tw.sum(-1, keepdim=True)).cuda()
        self.activated = int(torch.unique(self.ti).numel())

    def enqueue_one(self):
        logits = torch.nn.functional.linear(self.h.float(),
                                            self.gate_w.float())
        torch.topk(torch.softmax(logits, dim=-1), self.top_k, dim=-1)
        return moe_kernels.run_experts(self.kernel, self.h, self.w13,
                                       self.w2, self.tw, self.ti, self.E)


class PrefillAttnDriver:
    """One prefill attention sublayer (QKV projection, QK norm, RoPE,
    FlashAttention-3, output projection) over L tokens. Several input and
    cache sets rotate so the working set does not stay in the L2."""

    def __init__(self, name, model, layer, L, seed, n_rot=4):
        self.name = name
        self.layer = layer
        self.n_rot = n_rot
        cfg = model.config
        self.cpos = torch.arange(L, device="cuda")
        probe = torch.zeros(1, L, cfg.hidden_size, dtype=torch.bfloat16,
                            device="cuda")
        self.pos_emb = model.model.rotary_emb(probe, self.cpos[None])
        self.sets = []
        for r in range(n_rot):
            g = torch.Generator(device="cpu").manual_seed(seed + r)
            h = (torch.randn(1, L, cfg.hidden_size, generator=g)
                 * 0.5).to(torch.bfloat16).cuda()
            cache = mh.FlashStaticCache(1, L + 8, cfg.num_key_value_heads,
                                        cfg.head_dim, torch.bfloat16, "cuda")
            self.sets.append((h, cache))
        self.i = 0

    def rotate(self):
        self.i = (self.i + 1) % self.n_rot

    def enqueue_one(self):
        h, cache = self.sets[self.i]
        cache.reset_to(0)
        self.layer.self_attn(hidden_states=h, position_embeddings=self.pos_emb,
                             attention_mask=None, past_key_values=cache,
                             cache_position=self.cpos)


@triton.jit
def _sector_walk(ptr, n_elem, out_ptr, BLOCK: tl.constexpr):
    """Each row loads one 32-byte sector (16 bf16 values); rows sit 128
    bytes apart, so every sector is its own L2 request, the request size
    the streaming expert GEMMs issue."""
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * BLOCK * 64
    offs = base + tl.arange(0, BLOCK).to(tl.int64) * 64
    idx = offs[:, None] + tl.arange(0, 16)[None, :]
    v = tl.load(ptr + idx, mask=idx < n_elem, other=0.0)
    tl.store(out_ptr + pid, tl.sum(v.to(tl.float32)))


class SectorReaderDriver:
    """Synthetic expert reader: one unit walks `unit_layers` layers of expert
    weights (w13 and w2) sector by sector, wrapping around the list. The
    kernel, grid, and layers are identical for both placements; only the
    base pointers differ (HBM tensors or pinned CPU DRAM views)."""

    BLOCK = 256

    def __init__(self, name, flats, unit_layers=8):
        self.name = name
        self.flats = flats
        self.outs = [torch.empty(triton.cdiv(t.numel(), self.BLOCK * 64),
                                 dtype=torch.float32, device="cuda")
                     for t in flats]
        self.unit = unit_layers * 2
        self.i = 0
        # A unit loads a quarter of each tensor (32 B of every 128 B).
        self.unit_bytes = sum(t.numel() * t.element_size() // 4
                              for t in flats[:self.unit])
        self.activated = 0

    def enqueue_one(self):
        for _ in range(self.unit):
            t, o = self.flats[self.i], self.outs[self.i]
            _sector_walk[(o.numel(),)](t, t.numel(), o, BLOCK=self.BLOCK)
            self.i = (self.i + 1) % len(self.flats)


class CopyEngineDriver:
    """Host-to-device copies through the copy engine over the first
    n_chunks chunks of a pinned pool; one unit is per_unit chunk copies
    into one reused device buffer."""

    def __init__(self, name, chunks, n_chunks, dst, per_unit=4):
        self.name = name
        self.chunks = chunks[:n_chunks]
        self.dst = dst
        self.per_unit = per_unit
        self.i = 0
        self.unit_bytes = per_unit * dst.numel()
        self.activated = 0

    def enqueue_one(self):
        for _ in range(self.per_unit):
            self.dst.copy_(self.chunks[self.i], non_blocking=True)
            self.i = (self.i + 1) % len(self.chunks)


# ---------------------------------------------------------------------------
# Instrument
# ---------------------------------------------------------------------------
class ChainRunner:
    """k driver iterations captured into one CUDA graph. A few dozen
    replays hold seconds of pending work on the contender's partition,
    which pre-enqueued individual launches cannot: the driver's pushbuffer
    only holds a bounded backlog."""

    def __init__(self, mod, part_idx, driver, unit_ms, target_ms=50.0):
        self.name = driver.name
        self.mod, self.part_idx = mod, part_idx
        self.k = max(1, min(1000, int(round(target_ms / max(unit_ms, 1e-3)))))
        with partitions.on_partition(mod, part_idx) as s:
            with torch.cuda.stream(s):
                driver.enqueue_one()
            s.synchronize()
            torch.cuda.empty_cache()
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g, stream=s):
                for _ in range(self.k):
                    driver.enqueue_one()
                    if hasattr(driver, "rotate"):
                        driver.rotate()
            s.synchronize()

    def enqueue_one(self):
        self.g.replay()

    def close(self):
        # A graph must be released before its green context is destroyed.
        self.mod.partition_push(self.part_idx)
        try:
            self.g.reset()
        finally:
            self.mod.partition_pop()


class EagerRotator:
    def __init__(self, drv):
        self.drv = drv
        self.name = drv.name

    def enqueue_one(self):
        self.drv.enqueue_one()
        if hasattr(self.drv, "rotate"):
            self.drv.rotate()


def timed_iters(mod, part_idx, driver, n, warm, spin_s=0.2, return_evs=False):
    """[spin, warm iterations, n timed iterations] on the partition's
    stream; returns the per-iteration times in ms."""
    with partitions.on_partition(mod, part_idx) as s:
        evs = [(torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True)) for _ in range(n)]
        with torch.cuda.stream(s):
            torch.cuda._sleep(sleep_cycles(spin_s))
            for _ in range(warm):
                driver.enqueue_one()
            for e0, e1 in evs:
                e0.record(s)
                driver.enqueue_one()
                e1.record(s)
        s.synchronize()
    ts = [e0.elapsed_time(e1) for e0, e1 in evs]
    return (ts, evs) if return_evs else ts


def enqueue_chain(mod, part_idx, runner, iters):
    with partitions.on_partition(mod, part_idx) as s:
        with torch.cuda.stream(s):
            for _ in range(iters):
                runner.enqueue_one()
            end = torch.cuda.Event(enable_timing=True)
            end.record(s)
    return s, end


def warm_eager(mod, part_idx, driver, iters=3):
    """Absorb the per-context lazy module loads and settle the caching
    allocator before an eager measurement."""
    with partitions.on_partition(mod, part_idx) as s:
        with torch.cuda.stream(s):
            for _ in range(iters):
                driver.enqueue_one()
        s.synchronize()


def run_cell(mod, out_f, cell, side_a, side_b, sm_a, sm_b, n=20, warm=1):
    sizes = partitions.create(mod, [sm_a, sm_b])
    log(f"[{cell}] partitions {sizes}")
    raw = (EagerRotator(side_a), EagerRotator(side_b))
    for idx in (0, 1):
        warm_eager(mod, idx, raw[idx])
    solo = {}
    results = []
    for idx in (0, 1):
        ts = timed_iters(mod, idx, raw[idx], n, warm)
        solo[raw[idx].name] = statistics.median(ts)
        log(f"  solo  {raw[idx].name:16s}: median {solo[raw[idx].name]:8.3f}"
            f" ms")
        results.append(dict(cell=cell, side=raw[idx].name, mode="solo",
                            sm=sizes[idx], partner=None, times_ms=ts))
    for meas_idx, cont_idx in ((1, 0), (0, 1)):
        meas, cont = raw[meas_idx], raw[cont_idx]
        window = 0.2 + (warm + n) * solo[meas.name] / 1e3
        unit_s = solo[cont.name] / 1e3
        chain = ChainRunner(mod, cont_idx, cont.drv, solo[cont.name])
        reps = int(window * 3.0 / (chain.k * unit_s)) + 2
        for attempt in range(5):
            cs, cend = enqueue_chain(mod, cont_idx, chain, reps)
            ts, evs = timed_iters(mod, meas_idx, meas, n, warm,
                                  return_evs=True)
            cs.synchronize()
            margin = evs[-1][1].elapsed_time(cend)   # > 0: contender outlived
            if margin > 0 or attempt == 4 or reps >= 2000:
                break
            reps = min(reps * 4, 2000)
            log(f"    contender drained early; retry with {reps} replays")
        deg = statistics.median(ts) / solo[meas.name]
        log(f"  corun {meas.name:16s} beside {cont.name}: x{deg:.3f}"
            f"{'' if margin > 0 else '  [contender did not cover]'}")
        results.append(dict(cell=cell, side=meas.name, mode="corun",
                            sm=sizes[meas_idx], partner=cont.name,
                            partner_sm=sizes[cont_idx], chain_k=chain.k,
                            chain_reps=reps, covered=bool(margin > 0),
                            margin_ms=margin, times_ms=ts))
        chain.close()
    for r in results:
        for d in (side_a, side_b):
            if r["side"] == d.name and hasattr(d, "unit_bytes"):
                r["unit_bytes"] = d.unit_bytes
        out_f.write(json.dumps(r) + "\n")
    out_f.flush()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    mod.partitions_destroy()


# ---------------------------------------------------------------------------
def load_layer(model_path, layer_idx):
    """Full model with FlashAttention-3 and fused MoE blocks; returns the
    model, one layer (its attention indexes a single-layer cache), and that
    layer's router and expert weights in HBM and in pinned CPU DRAM."""
    model = mh.load_stock_model(model_path)
    mh.use_fa3(model)
    mh.convert_moe_layers(model)
    layer = model.model.layers[layer_idx]
    layer.self_attn.layer_idx = 0
    blk = layer.mlp
    keep13, w13_cpu = to_pinned_cuda_view(blk.w13)
    keep2, w2_cpu = to_pinned_cuda_view(blk.w2)
    return dict(model=model, layer=layer, gate_w=blk.gate.weight.detach(),
                w13_hbm=blk.w13, w2_hbm=blk.w2, w13_cpu=w13_cpu,
                w2_cpu=w2_cpu, keep=(keep13, keep2))


@torch.inference_mode()
def main():
    # inference_mode matters here: under autograd every in-place KV-cache
    # write would keep its source tensors alive and eager chains would
    # accumulate memory.
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--suite", choices=("pairs", "reader", "copy-engine"),
                    required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--n", type=int, default=20,
                    help="timed iterations per measurement")
    ap.add_argument("--share", type=float, default=0.48,
                    help="SM share of each side in the pairs suite and of "
                         "the attention victim in the other suites")
    ap.add_argument("--reader-sms", type=int, nargs="+",
                    default=[8, 16, 24, 32, 48, 64, 68])
    ap.add_argument("--reader-layers", type=int, default=16)
    ap.add_argument("--footprints-gb", type=int, nargs="+",
                    default=[1, 4, 8, 16, 32, 64, 128])
    args = ap.parse_args()

    total = torch.cuda.get_device_properties(0).multi_processor_count
    # Partition sizes are rounded to multiples of 8 SMs (64 of 132 for 48%).
    sm_v = max(8, 8 * round(total * args.share / 8))
    log(f"GPU: {torch.cuda.get_device_properties(0).name}, {total} SMs; "
        f"victim partition {sm_v} SMs")
    mod = partitions.build()
    L = load_layer(args.model_path, args.layer)
    model, cfg = L["model"], L["model"].config
    E, top_k = cfg.num_experts, cfg.num_experts_per_tok
    attn = PrefillAttnDriver("attention", model, L["layer"], 4096, 301)

    with open(args.out, "a") as f:
        if args.suite == "pairs":
            def moe(name, kernel, seed):
                return MoeDriver(name, L["gate_w"], L["w13_cpu"],
                                 L["w2_cpu"], 4096, kernel, top_k, E, seed,
                                 N_ACTIVE_PREFILL_4K)
            attn2 = PrefillAttnDriver("attention-2", model, L["layer"], 4096,
                                      501)
            sym, sym2 = moe("sym", "sym", 101), moe("sym-2", "sym", 151)
            asym, asym2 = moe("asym", "asym", 101), moe("asym-2", "asym", 151)
            for d in (attn, attn2, sym, sym2, asym, asym2):
                d.enqueue_one()
            torch.cuda.synchronize()
            log(f"experts activated per layer: {sym.activated}")
            for cell, a, b in (("attn+attn", attn, attn2),
                               ("sym+sym", sym, sym2),
                               ("asym+asym", asym, asym2),
                               ("sym+asym", sym2, asym),
                               ("attn+sym", attn, sym),
                               ("attn+asym", attn, asym)):
                run_cell(mod, f, cell, a, b, sm_v, sm_v, n=args.n)

        elif args.suite == "reader":
            hbm_flats, cpu_flats, keep = [], [], []
            for i in range(args.reader_layers):
                blk = model.model.layers[i].mlp
                k13, v13 = to_pinned_cuda_view(blk.w13)
                k2, v2 = to_pinned_cuda_view(blk.w2)
                keep.append((k13, k2))
                hbm_flats += [blk.w13.view(-1), blk.w2.view(-1)]
                cpu_flats += [v13.view(-1), v2.view(-1)]
            span = sum(t.numel() * 2 for t in hbm_flats) / 1e9
            log(f"reader footprint: {args.reader_layers} layers, "
                f"{span:.1f} GB per placement")
            readers = {"cpu": SectorReaderDriver("reader-cpu", cpu_flats),
                       "hbm": SectorReaderDriver("reader-hbm", hbm_flats)}
            for d in (attn, *readers.values()):
                d.enqueue_one()
            torch.cuda.synchronize()
            for s in args.reader_sms:
                if s + sm_v > total:
                    log(f"skip {s} reader SMs: exceeds the device")
                    continue
                for place, rd in readers.items():
                    run_cell(mod, f, f"attn+reader-{place}-{s}", attn, rd,
                             sm_v, s, n=args.n)

        else:
            chunk_mb = 256
            n_total = max(args.footprints_gb) * 1024 // chunk_mb
            log(f"pinning {max(args.footprints_gb)} GB in {chunk_mb} MB "
                f"chunks")
            # Page-locked by registration, not by torch's pinned allocator:
            # the copies run on partition streams that are destroyed after
            # every cell, and the allocator would record events on those
            # streams when it frees the pool.
            pool = torch.empty(n_total * chunk_mb * 2**20, dtype=torch.uint8)
            cuda_check(torch.cuda.cudart().cudaHostRegister(
                pool.data_ptr(), pool.numel(), 0), "cudaHostRegister")
            chunks = list(pool.view(n_total, -1))
            dst = torch.empty(chunk_mb * 2**20, dtype=torch.uint8,
                              device="cuda")
            attn.enqueue_one()
            dst.copy_(chunks[0], non_blocking=True)
            torch.cuda.synchronize()
            # The contender partition only hosts the copy stream; the copy
            # engine itself uses no SMs.
            sm_ce = max(8, 8 * round(total * 0.24 / 8))
            for gb in args.footprints_gb:
                ce = CopyEngineDriver(f"copy-engine-{gb}GB", chunks,
                                      gb * 1024 // chunk_mb, dst)
                run_cell(mod, f, f"attn+copy-engine-{gb}GB", attn, ce, sm_v,
                         sm_ce, n=args.n)
            torch.cuda.synchronize()
            cuda_check(torch.cuda.cudart().cudaHostUnregister(
                pool.data_ptr()), "cudaHostUnregister")
    log(f"suite {args.suite} done")


if __name__ == "__main__":
    main()
