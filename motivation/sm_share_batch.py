"""Batched decode latency versus the SM share (Figs. 13 and 14).

Every request has a 1K-token prompt; the batch decodes 128 steps together
inside a green context of s SMs, and the attention and MoE spans are summed
over all layers and steps exactly as in sm_share.py.

Fig. 13 uses batch sizes 8 to 128 with the symmetric kernel and a 64-row
tile. Fig. 14 compares the symmetric kernel, whose tile height follows the
batch (--sym-block-m auto), with the asymmetric kernel at batch sizes 1 to
256.

    python sm_share_batch.py --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --batches 8 16 32 64 128 --moe-kernel sym --out results/batch.jsonl
"""
import argparse
import gc
import json

import torch

import model_harness as mh
from common import (GreenCtx, ModuleTimer, burn_in, cos, kept_spread, log,
                    sleep_cycles, trimmed_summarize)

VAL_B, VAL_L, VAL_STEPS = 4, 64, 4


@torch.inference_mode()
def build_reference(model, cfg):
    """Greedy decode with the stock model (batched SDPA, dynamic cache)."""
    from transformers.cache_utils import DynamicCache
    prompts = [mh.make_prompt(VAL_L, cfg.vocab_size, seed=5000 + j)
               for j in range(VAL_B)]
    c = DynamicCache()
    out = model(input_ids=torch.cat(prompts, dim=0), past_key_values=c,
                use_cache=True)
    tok = out.logits[:, -1:, :].argmax(dim=-1)
    logits, inputs = [], []
    for _ in range(VAL_STEPS):
        inputs.append(tok)
        out = model(input_ids=tok, past_key_values=c, use_cache=True)
        logits.append(out.logits[:, -1, :].float())
        tok = out.logits[:, -1:, :].argmax(dim=-1)
    return prompts, logits, inputs


@torch.inference_mode()
def validate(model, cfg, ref):
    """Replay the reference's input tokens through the measured
    configuration and compare the logits step by step."""
    prompts, ref_logits, ref_inputs = ref
    cache = mh.BatchStridedCache(cfg.num_hidden_layers, VAL_B, VAL_L + 16,
                                 cfg.num_key_value_heads, cfg.head_dim,
                                 torch.bfloat16, "cuda")
    mh.prefill_all(model, cache, prompts)
    for i in range(VAL_STEPS):
        out = model(input_ids=ref_inputs[i], past_key_values=cache,
                    use_cache=True,
                    cache_position=torch.arange(VAL_L + i, VAL_L + i + 1,
                                                device="cuda"))
        c = cos(out.logits[:, -1, :].float(), ref_logits[i])
        log(f"validation: decode step {i} logits cos={c:.6f}")
        assert c > 0.99
    del cache
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batches", type=int, nargs="+",
                   default=[8, 16, 32, 64, 128])
    p.add_argument("--sms", type=int, nargs="+",
                   default=[8, 16, 24, 32, 48, 64, 96, 132])
    p.add_argument("--prompt-len", type=int, default=1024)
    p.add_argument("--decode-steps", type=int, default=128)
    p.add_argument("--decode-runs", type=int, default=8)
    p.add_argument("--experts-in", choices=("cpu", "hbm"), default="cpu")
    p.add_argument("--moe-kernel", choices=("sym", "asym"), default="sym")
    p.add_argument("--sym-block-m", default="64",
                   help="symmetric tile height, or auto for the smallest "
                        "height covering the batch")
    p.add_argument("--moe-spacer-us", type=float, default=500.0)
    p.add_argument("--attn-spacer-us", type=float, default=500.0)
    p.add_argument("--skip-validate", action="store_true")
    p.add_argument("--free-running", dest="replay", action="store_false",
                   help="greedy decoding instead of fixed-token replay")
    args = p.parse_args()

    total = torch.cuda.get_device_properties(0).multi_processor_count
    sms = sorted({s for s in args.sms if s < total} | {total})
    L = args.prompt_len
    torch.manual_seed(0)
    log(f"GPU: {torch.cuda.get_device_properties(0).name}, {total} SMs")

    model = mh.load_stock_model(args.model_path)
    cfg = model.config
    ref = None if args.skip_validate else build_reference(model, cfg)
    mh.prepare(model, args.experts_in, args.moe_kernel, args.sym_block_m)
    if ref is not None:
        validate(model, cfg, ref)
        log("=== validation passed ===")
    mh.attach_spacers(model, args.moe_spacer_us, args.attn_spacer_us)
    timer = ModuleTimer(model)
    fwd_ev = (torch.cuda.Event(enable_timing=True),
              torch.cuda.Event(enable_timing=True))

    done = set()
    try:
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add((r["B"], r["sm"]))
            except (ValueError, KeyError):
                pass
        log(f"resuming: {len(done)} points already present")
    except FileNotFoundError:
        pass
    outf = open(args.out, "a", buffering=1)

    burn_in(20)
    dec = sleep_cycles(0.2)
    replay_all = None
    if args.replay:
        n_cols = max(args.decode_steps, 128) + 8
        replay_all = torch.cat(
            [mh.make_prompt(n_cols, cfg.vocab_size, seed=7000 + j)
             for j in range(max(args.batches))], dim=0)

    def new_cache(B, steps):
        return mh.BatchStridedCache(cfg.num_hidden_layers, B, L + steps + 12,
                                    cfg.num_key_value_heads, cfg.head_dim,
                                    torch.bfloat16, "cuda")

    with torch.inference_mode():
        warm_b = min(4, max(args.batches))
        cache = new_cache(warm_b, 128)   # the warmup decodes 128 steps
        tok0 = mh.prefill_all(model, cache, [
            mh.make_prompt(L, cfg.vocab_size, seed=1000 + j)
            for j in range(warm_b)], timer)
        rp = replay_all[:warm_b] if replay_all is not None else None
        for _ in range(2):
            mh.decode_batch(model, timer, cache, tok0, L, 128, decouple=dec,
                            replay=rp)
            torch.cuda.synchronize()
        del cache
        gc.collect()
        torch.cuda.empty_cache()

        for B in args.batches:
            cache = new_cache(B, max(8, args.decode_steps))
            tok0 = mh.prefill_all(model, cache, [
                mh.make_prompt(L, cfg.vocab_size, seed=1000 + j)
                for j in range(B)], timer)
            rp = replay_all[:B] if replay_all is not None else None
            log(f"B={B}: {B} prompts of {L} tokens prefilled")
            for sm in sms:
                if (B, sm) in done:
                    continue
                pct = 100.0 * sm / total
                with GreenCtx(sm, total):
                    mh.decode_batch(model, timer, cache, tok0, L, 8,
                                    decouple=dec, replay=rp)
                    torch.cuda.synchronize()
                    best = None
                    for _ in range(3):
                        attn_l, moe_l, fwd_l = [], [], []
                        for _ in range(args.decode_runs):
                            mh.decode_batch(model, timer, cache, tok0, L,
                                            args.decode_steps, fwd_ev,
                                            decouple=dec, replay=rp)
                            torch.cuda.synchronize()
                            t = timer.collect()
                            attn_l.append(t["attn"])
                            moe_l.append(t["moe"])
                            fwd_l.append(fwd_ev[0].elapsed_time(fwd_ev[1]))
                        spread = (max(kept_spread(attn_l),
                                      kept_spread(moe_l))
                                  if args.decode_runs > 1 else 0.0)
                        if best is None or spread < best[0]:
                            best = (spread, attn_l, moe_l, fwd_l)
                        if spread <= 0.05:
                            break
                    spread, attn_l, moe_l, fwd_l = best
                    rec = {"phase": "decode", "B": B, "L": L, "sm": sm,
                           "sm_pct": pct, "decode_steps": args.decode_steps,
                           "experts_in": args.experts_in,
                           "moe_kernel": args.moe_kernel,
                           "sym_block_m": args.sym_block_m,
                           "workload": ("fixed-token-replay" if args.replay
                                        else "free-running-greedy"),
                           "kept_spread": spread,
                           "attn_ms": trimmed_summarize(attn_l),
                           "moe_ms": trimmed_summarize(moe_l),
                           "fwd_ms": trimmed_summarize(fwd_l)}
                    outf.write(json.dumps(rec) + "\n")
                    log(f"decode B={B} sm={sm} ({pct:.0f}%): attention "
                        f"{rec['attn_ms']['value']:.1f} ms, MoE "
                        f"{rec['moe_ms']['value']:.1f} ms")
            del cache
            gc.collect()
            torch.cuda.empty_cache()
    outf.close()
    log("=== batch sweep complete ===")


if __name__ == "__main__":
    main()
