"""Expert activation profiles, measured by serving sample requests.

The decode profile, per layer the share of the decode batch's routed rows
each expert receives, seeds the hot tier and the expected misses admission
plans with. The prefill profile orders the experts the staging buffers
take. The two are kept apart because one long prompt activates nearly
every expert and would swamp the decode ranking.

    python -m monoserve.profile --model /path/to/model --calibration calibration.json \\
        --prompts prompts.jsonl --out profile.json

prompts.jsonl holds one request per line: {"prompt": text} or
{"prompt_token_ids": [...]}, optionally with "max_tokens".
"""
import argparse
import json

import numpy as np

from monoserve.deploy import DeployOptions, build_engine


def shares(counts):
    """Per layer, each expert's share of the routed rows (uniform where a
    layer saw none)."""
    c = np.asarray(counts, dtype=np.float64)
    total = c.sum(axis=1, keepdims=True)
    return np.where(total > 0, c / np.maximum(total, 1.0), 1.0 / c.shape[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256, help="requests to serve")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--kv-gb", type=float, default=None)
    ap.add_argument("--hbm-budget-gb", type=float, default=None)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    reqs = []
    with open(args.prompts) as f:
        for line in f:
            d = json.loads(line)
            ids = d.get("prompt_token_ids") or tok(d["prompt"])["input_ids"]
            max_tokens = int(d.get("max_tokens", args.max_tokens))
            reqs.append((ids[:args.max_len - max_tokens - 1], max_tokens))
            if len(reqs) >= args.n:
                break
    opts = DeployOptions(calibration=args.calibration, kv_gb=args.kv_gb,
                         hbm_budget_gb=args.hbm_budget_gb, max_len=args.max_len)
    eng = build_engine(args.model, opts)
    eos = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    eng.start()
    try:
        pending = set(range(len(reqs)))
        for i, (ids, max_tokens) in enumerate(reqs):
            # targets loose enough that admission never rejects a request
            for rid, _, _ in eng.add_request(i, ids, max_tokens=max_tokens, eos=eos, ttft=3600.0,
                                             tpot=60.0):
                pending.discard(rid)
        while pending:
            for rid, _, reason in eng.step(wait=0.01):
                if reason:
                    pending.discard(rid)
        while not eng.idle():
            eng.step(wait=0.01)
    finally:
        eng.stop()
    decode = sum(lane.hist.cpu().numpy() for lane in eng.decode_lanes)
    prefill = sum(lane.hist.cpu().numpy() for lane in eng.prefill_lanes)
    with open(args.out, "w") as f:
        json.dump({"decode": shares(decode).tolist(), "prefill": shares(prefill).tolist(),
                   "requests": len(reqs)}, f)
    print(f"wrote {args.out} from {len(reqs)} requests")


if __name__ == "__main__":
    main()
