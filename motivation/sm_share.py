"""Latency of attention and MoE versus the SM share (Figs. 2 and 4).

For every input length L and SM count s, the model runs one prefill pass
of L tokens and 128 decode steps at batch size 1 inside a green context
that confines all kernels to s SMs. The spans of all attention sublayers
(QKV projection, QK norm, RoPE, FlashAttention-3, output projection) and of
all MoE sublayers (router and expert FFNs) are summed over the 48 layers.

Fig. 2 uses --experts-in cpu --moe-kernel sym. Fig. 4 adds the prefill MoE
spans of --moe-kernel asym (--skip-decode suffices).

    python sm_share.py --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --moe-kernel sym --out results/sm_share_sym.jsonl

Output: one JSON record per (phase, L, SM count) with the per-iteration
sums; resumes if --out already holds some points.
"""
import argparse
import gc
import json
import os
import time

import torch

import model_harness as mh
from common import (GreenCtx, ModuleTimer, NullTimer, SM_CLOCK_HZ, burn_in,
                    cos, kept_spread, log, sleep_cycles, summarize,
                    trimmed_summarize)


def calibrate_decouple(enqueue_fn, sec=0.20):
    """Report the CPU enqueue time of one unit of work and return the spin
    that precedes each measured region (fixed at 200 ms, which covered the
    enqueue of every point with a wide margin)."""
    torch.cuda.synchronize()
    torch.cuda._sleep(sleep_cycles(1.0))
    t0 = time.perf_counter()
    enqueue_fn()
    enq = time.perf_counter() - t0
    torch.cuda.synchronize()
    log(f"decouple: enqueue {enq * 1e3:.1f} ms, spin {sec * 1e3:.0f} ms")
    return sleep_cycles(sec)


@torch.inference_mode()
def validate(args):
    """Check the substituted kernels against the stock model before
    measuring anything."""
    log("=== validation ===")
    model = mh.load_stock_model(args.model_path)
    cfg = model.config
    hq, hk, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim

    for sq, sk, name in ((512, 512, "prefill"), (1, 512, "decode")):
        q = torch.randn(1, hq, sq, d, dtype=torch.bfloat16, device="cuda") * .1
        k = torch.randn(1, hk, sk, d, dtype=torch.bfloat16, device="cuda") * .1
        v = torch.randn(1, hk, sk, d, dtype=torch.bfloat16, device="cuda") * .1
        o_fa, _ = mh.fa3_attention(None, q, k, v, scaling=d ** -0.5)
        o_ref = torch.nn.functional.scaled_dot_product_attention(
            q.float(), k.float().repeat_interleave(hq // hk, dim=1),
            v.float().repeat_interleave(hq // hk, dim=1),
            is_causal=(sq == sk), scale=d ** -0.5).transpose(1, 2)
        c = cos(o_fa, o_ref)
        log(f"FlashAttention-3 vs SDPA ({name}): cos={c:.6f}")
        assert c > 0.999

    ids = mh.make_prompt(64, cfg.vocab_size, seed=1234)
    ref = model(input_ids=ids).logits[:, -1, :].float()
    blk = model.model.layers[0].mlp
    x = torch.randn(1, 64, cfg.hidden_size, dtype=torch.bfloat16,
                    device="cuda") * 0.5
    ref_moe = blk(x)
    ref_moe = ref_moe[0] if isinstance(ref_moe, tuple) else ref_moe
    c = cos(mh.FusedMoeBlock(blk)(x), ref_moe)
    log(f"fused MoE block vs stock block (layer 0): cos={c:.6f}")
    assert c > 0.99

    # FlashAttention-3 and fused MoE blocks with the experts still in HBM:
    # the configuration whose compute-bound prefill checks the green context.
    mh.prepare(model, "hbm", args.moe_kernel, args.sym_block_m)
    cache = mh.FlashStaticCache(cfg.num_hidden_layers, 4096,
                                cfg.num_key_value_heads, cfg.head_dim,
                                torch.bfloat16, "cuda")
    out = mh.run_prefill_once(model, NullTimer(), cache, ids)
    c = cos(out.logits[:, -1, :].float(), ref)
    log(f"FlashAttention-3 + fused MoE (experts in HBM) vs stock model, "
        f"last-token logits: cos={c:.6f}")
    assert c > 0.99

    total = torch.cuda.get_device_properties(0).multi_processor_count
    ids4k = mh.make_prompt(4096, cfg.vocab_size, seed=42)
    big = mh.FlashStaticCache(cfg.num_hidden_layers, 8192,
                              cfg.num_key_value_heads, cfg.head_dim,
                              torch.bfloat16, "cuda")

    def timed():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mh.run_prefill_once(model, NullTimer(), big, ids4k)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    timed()
    t_full = min(timed() for _ in range(2))
    with GreenCtx(16, total):
        timed()
        t_16 = min(timed() for _ in range(2))
    log(f"green context check: 4K prefill {t_full * 1e3:.1f} ms on all SMs, "
        f"{t_16 * 1e3:.1f} ms on 16 SMs ({t_16 / t_full:.2f}x)")
    assert t_16 / t_full > 2.0, "green context does not seem to limit SMs"

    torch.cuda.reset_peak_memory_stats()
    ids16k = mh.make_prompt(16384, cfg.vocab_size, seed=7)
    c16 = mh.FlashStaticCache(cfg.num_hidden_layers, 17000,
                              cfg.num_key_value_heads, cfg.head_dim,
                              torch.bfloat16, "cuda")
    mh.MASK_SEEN[0] = False
    mh.run_prefill_once(model, NullTimer(), c16, ids16k)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    log(f"16K prefill peak memory {peak:.1f} GiB, mask seen: "
        f"{mh.MASK_SEEN[0]}")
    assert not mh.MASK_SEEN[0], "transformers materialized an attention mask"

    if args.experts_in == "cpu":
        mh.move_experts_to_cpu(model, args.moe_kernel)
        out = mh.run_prefill_once(model, NullTimer(), cache, ids)
        c = cos(out.logits[:, -1, :].float(), ref)
        log(f"experts in CPU DRAM with the {args.moe_kernel} kernel vs stock "
            f"model, last-token logits: cos={c:.6f}")
        assert c > 0.99
    log("=== validation passed ===")
    del model, cache, big, c16
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def sweep(args):
    log("=== sweep ===")
    total = torch.cuda.get_device_properties(0).multi_processor_count
    sms = sorted({s for s in args.sms if s < total} | {total})
    lengths = list(args.lengths)

    model = mh.load_stock_model(args.model_path)
    mh.prepare(model, args.experts_in, args.moe_kernel, args.sym_block_m)
    cfg = model.config
    mh.attach_spacers(model, args.moe_spacer_us, args.attn_spacer_us)
    timer = ModuleTimer(model)
    fwd_ev = (torch.cuda.Event(enable_timing=True),
              torch.cuda.Event(enable_timing=True))

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add((r["phase"], r["L"], r["sm"]))
            except (ValueError, KeyError):
                pass
        log(f"resuming: {len(done)} points already in {args.out}")
    outf = open(args.out, "a", buffering=1)

    def emit(rec):
        rec.update(experts_in=args.experts_in, moe_kernel=args.moe_kernel,
                   sym_block_m=args.sym_block_m)
        outf.write(json.dumps(rec) + "\n")

    def new_cache(L):
        return mh.FlashStaticCache(cfg.num_hidden_layers,
                                   L + args.decode_steps + 8,
                                   cfg.num_key_value_heads, cfg.head_dim,
                                   torch.bfloat16, "cuda")

    burn_in(20)
    log("warmup: compile every shape once on all SMs")
    for L in list(lengths):
        try:
            ids = mh.make_prompt(L, cfg.vocab_size, seed=100 + L)
            cache = new_cache(L)
            out = mh.run_prefill_once(model, timer, cache, ids)
            tok = out.logits[:, -1:, :].argmax(dim=-1)
            mh.run_decode_steps(model, timer, cache, tok, L, 4)
            torch.cuda.synchronize()
            del cache
        except torch.cuda.OutOfMemoryError:
            log(f"L={L}: out of memory during warmup, dropping this length")
            lengths.remove(L)
        gc.collect()
        torch.cuda.empty_cache()

    dec = 0
    if args.decouple:
        ids_c = mh.make_prompt(1024, cfg.vocab_size, seed=99)
        cache_c = mh.FlashStaticCache(cfg.num_hidden_layers, 1024 + 140,
                                      cfg.num_key_value_heads, cfg.head_dim,
                                      torch.bfloat16, "cuda")
        out_c = mh.run_prefill_once(model, timer, cache_c, ids_c)
        tok_c = out_c.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()
        dec = calibrate_decouple(lambda: mh.run_decode_steps(
            model, timer, cache_c, tok_c, 1024, 1))
        for _ in range(2):   # settle clocks on the decode path
            mh.run_decode_steps(model, timer, cache_c, tok_c, 1024, 128,
                                decouple=dec)
            torch.cuda.synchronize()
        del cache_c
        gc.collect()
        torch.cuda.empty_cache()
    sleep_ms = dec / SM_CLOCK_HZ * 1e3

    for L in lengths:
        ids = mh.make_prompt(L, cfg.vocab_size, seed=100 + L)
        cache = new_cache(L)
        # The cache contents do not depend on the SM count: fill it once.
        out = mh.run_prefill_once(model, timer, cache, ids)
        tok = out.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()
        for sm in sms:
            pct = 100.0 * sm / total
            with GreenCtx(sm, total):
                if ("prefill", L, sm) not in done:
                    for _ in range(2):
                        mh.run_prefill_once(model, timer, cache, ids,
                                            decouple=dec)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    mh.run_prefill_once(model, timer, cache, ids,
                                        decouple=dec)
                    torch.cuda.synchronize()
                    est = time.perf_counter() - t0
                    iters = args.prefill_iters or (5 if est < 1.5 else 3)
                    attn_l, moe_l, fwd_l = [], [], []
                    for _ in range(iters):
                        mh.run_prefill_once(model, timer, cache, ids, fwd_ev,
                                            decouple=dec)
                        torch.cuda.synchronize()
                        t = timer.collect()
                        attn_l.append(t["attn"])
                        moe_l.append(t["moe"])
                        fwd_l.append(fwd_ev[0].elapsed_time(fwd_ev[1]))
                    emit({"phase": "prefill", "L": L, "sm": sm,
                          "sm_pct": pct, "decoupled": bool(dec),
                          "sleep_ms": sleep_ms,
                          "attn_ms": summarize(attn_l),
                          "moe_ms": summarize(moe_l),
                          "fwd_ms": summarize(fwd_l)})
                    log(f"prefill L={L} sm={sm} ({pct:.0f}%): attention "
                        f"{sum(attn_l) / iters:.1f} ms, MoE "
                        f"{sum(moe_l) / iters:.1f} ms")

                if args.skip_decode or ("decode", L, sm) in done:
                    continue
                mh.run_decode_steps(model, timer, cache, tok, L, 8,
                                    decouple=dec)
                torch.cuda.synchronize()
                # Guard against a disturbed sample on a shared machine: if
                # the kept values still spread by more than 5%, measure the
                # point again (at most three times) and keep the tightest.
                best = None
                for _ in range(3):
                    attn_l, moe_l, fwd_l = [], [], []
                    for _ in range(args.decode_runs):
                        mh.run_decode_steps(model, timer, cache, tok, L,
                                            args.decode_steps, fwd_ev,
                                            decouple=dec)
                        torch.cuda.synchronize()
                        t = timer.collect()
                        attn_l.append(t["attn"])
                        moe_l.append(t["moe"])
                        fwd_l.append(fwd_ev[0].elapsed_time(fwd_ev[1]))
                    spread = (max(kept_spread(attn_l), kept_spread(moe_l))
                              if args.decode_runs > 1 else 0.0)
                    if best is None or spread < best[0]:
                        best = (spread, attn_l, moe_l, fwd_l)
                    if spread <= 0.05:
                        break
                    log(f"  kept spread {spread:.1%}, measuring again")
                spread, attn_l, moe_l, fwd_l = best
                rec = {"phase": "decode", "L": L, "sm": sm, "sm_pct": pct,
                       "decode_steps": args.decode_steps,
                       "decoupled": bool(dec), "sleep_ms": sleep_ms,
                       "kept_spread": spread,
                       "attn_ms": trimmed_summarize(attn_l),
                       "moe_ms": trimmed_summarize(moe_l),
                       "fwd_ms": trimmed_summarize(fwd_l)}
                emit(rec)
                log(f"decode  L={L} sm={sm} ({pct:.0f}%): attention "
                    f"{rec['attn_ms']['value']:.1f} ms, MoE "
                    f"{rec['moe_ms']['value']:.1f} ms")
        del cache
        gc.collect()
        torch.cuda.empty_cache()
    outf.close()
    log("=== sweep complete ===")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model-path", required=True)
    p.add_argument("--out", default="sm_share.jsonl")
    p.add_argument("--phase", choices=("validate", "sweep", "all"),
                   default="all")
    p.add_argument("--experts-in", choices=("cpu", "hbm"), default="cpu",
                   help="cpu: expert weights in pinned CPU DRAM, read by "
                        "--moe-kernel; hbm: weights in HBM, vLLM fused MoE")
    p.add_argument("--moe-kernel", choices=("sym", "asym"), default="sym")
    p.add_argument("--sym-block-m", default="64",
                   help="symmetric tile height, or auto")
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[1024, 2048, 4096, 8192, 16384, 32768, 65536,
                            131072])
    p.add_argument("--sms", type=int, nargs="+",
                   default=[8, 16, 24, 32, 48, 64, 96, 132])
    p.add_argument("--decode-steps", type=int, default=128)
    p.add_argument("--decode-runs", type=int, default=8,
                   help="repeats per decode point; the maximum and the "
                        "minimum are dropped before averaging")
    p.add_argument("--prefill-iters", type=int, default=0,
                   help="measured prefill passes per point (0: 5 for short "
                        "passes, 3 for long ones)")
    p.add_argument("--moe-spacer-us", type=float, default=500.0)
    p.add_argument("--attn-spacer-us", type=float, default=500.0)
    p.add_argument("--skip-decode", action="store_true")
    p.add_argument("--no-decouple", dest="decouple", action="store_false")
    args = p.parse_args()

    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    log(f"GPU: {props.name}, {props.multi_processor_count} SMs; "
        f"torch {torch.__version__}")
    if args.phase in ("validate", "all"):
        validate(args)
    if args.phase in ("sweep", "all"):
        sweep(args)


if __name__ == "__main__":
    main()
