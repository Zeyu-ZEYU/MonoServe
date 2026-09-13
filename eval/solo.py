"""Solo latency and the SLO targets derived from it.

Every request is served alone on one endpoint: requests go one at a time,
each after the previous one finished, through the same streaming client as
the sweeps. A request's solo TTFT and TPOT are the medians over --repeats
runs. Results are cached in <results>/<model>/<dataset>/solo.json keyed by
request id and written after every request, so a rerun skips the requests
already measured. The workload's reference request (1024 tokens) is
measured first; a few warm-up requests before it are not recorded.

A request meets its SLO when its TTFT is within alpha * max(its solo TTFT,
the reference's solo TTFT) and its TPOT within alpha * its solo TPOT
(targets()). Prompts under 1K tokens thus take the 1K prompt's TTFT target.
A request whose solo run produced fewer than two tokens takes the median
solo TPOT of the other requests.

Option for when per-request solo runs cost too much: --per-bucket K
measures only the first K requests (in workload order) of each prompt-length
bucket, and targets(..., mode="buckets") takes every request's solo TTFT and
TPOT from a piecewise-linear interpolation over prompt length through the
per-bucket medians (flat beyond the outermost buckets). metrics.py and the
figure scripts select this mode with --solo-mode buckets.

    python -m eval.solo --model qwen3-30b --dataset sharegpt \
        --endpoint http://<server>:8000
"""
import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

from .common import log, read_jsonl, solo_path, workload_path, write_json
from .metrics import finished, tpot
from .workloads import REFERENCE_ID

BUCKET_EDGES = [0] + [1024 * 2 ** i for i in range(9)]   # 0, 1K, ..., 256K


class Target(NamedTuple):
    ttft_ref: float           # the solo TTFT that sets the TTFT target (s)
    ttft: float               # TTFT target (s)
    tpot: Optional[float]     # TPOT target (s); None: no solo TPOT at all


def load_solo(path):
    path = Path(path)
    if not path.exists():
        return {"meta": {}, "solo": {}}
    with open(path) as f:
        return json.load(f)


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def bucket_of(n, edges=BUCKET_EDGES):
    return int(np.searchsorted(edges, n, side="right")) - 1


def select_for_buckets(requests, per_bucket, edges=BUCKET_EDGES):
    """The first per_bucket requests of every prompt-length bucket."""
    count, out = {}, []
    for r in requests:
        b = bucket_of(r["prompt_tokens"], edges)
        if count.get(b, 0) < per_bucket:
            count[b] = count.get(b, 0) + 1
            out.append(r)
    return out


def bucket_profile(entries, edges=BUCKET_EDGES):
    """Per-bucket medians of the measured requests: ((x, ttft), (x, tpot)),
    x being the median prompt length of the bucket."""
    groups = {}
    for e in entries:
        groups.setdefault(bucket_of(e["prompt_tokens"], edges), []).append(e)
    ft, fp = [], []
    for b in sorted(groups):
        g = groups[b]
        x = statistics.median(e["prompt_tokens"] for e in g)
        ft.append((x, statistics.median(e["ttft"] for e in g)))
        p = _median(e["tpot"] for e in g)
        if p is not None:
            fp.append((x, p))
    return tuple(tuple(map(list, zip(*pts))) if pts else ([], [])
                 for pts in (ft, fp))


def targets(requests, solo, alpha=5.0, *, mode="exact", edges=BUCKET_EDGES,
            reference=REFERENCE_ID):
    """{request id: Target} for the given requests. solo is the content of
    solo.json (or its "solo" mapping)."""
    entries = solo.get("solo", solo)
    measured = [e for k, e in entries.items() if k != reference]
    fallback_tpot = _median(e["tpot"] for e in measured)
    if mode == "exact":
        missing = [r["id"] for r in requests if r["id"] not in entries]
        if missing:
            raise KeyError(f"{len(missing)} requests have no solo latency "
                           f"(first: {missing[0]}); run eval.solo, or use "
                           f"the buckets mode")
        base = {r["id"]: (entries[r["id"]]["ttft"], entries[r["id"]]["tpot"])
                for r in requests}
    elif mode == "buckets":
        (xt, yt), (xp, yp) = bucket_profile(measured, edges)
        if not xt:
            raise KeyError("no solo latency measured")
        base = {r["id"]: (float(np.interp(r["prompt_tokens"], xt, yt)),
                          float(np.interp(r["prompt_tokens"], xp, yp))
                          if xp else None)
                for r in requests}
    else:
        raise ValueError(mode)
    if reference not in entries:
        raise KeyError("the reference request has no solo latency")
    floor = entries[reference]["ttft"]
    out = {}
    for rid, (ft, fp) in base.items():
        ref = max(ft, floor)
        fp = fallback_tpot if fp is None else fp
        out[rid] = Target(ref, alpha * ref, None if fp is None else alpha * fp)
    return out


async def profile(requests, endpoint, path, *, repeats=1, warmup=2,
                  model=None, sampling=None, continuous_usage=False,
                  api_key=None):
    """Measure every request not yet in the cache at path, one at a time."""
    from . import loadgen
    cache = load_solo(path)
    todo = [r for r in requests if r["id"] not in cache["solo"]]
    log(f"solo: {len(requests) - len(todo)} cached, {len(todo)} to measure")
    if not todo:
        return cache
    url = loadgen.base_url(endpoint) + "/v1/completions"
    async with loadgen.client_session(api_key) as session:
        if model is None:
            model = await loadgen.served_model(session, endpoint)
        cache["meta"].update(endpoint=endpoint, served_model=model,
                             repeats=repeats, sampling=sampling)

        async def once(req):
            rec = loadgen.new_record(req)
            body = loadgen.payload(req, model, sampling, continuous_usage)
            await loadgen.stream(session, url, body, time.perf_counter(), rec)
            return rec

        for _ in range(warmup):
            await once(requests[0])
        for i, req in enumerate(todo):
            runs = [await once(req) for _ in range(repeats)]
            bad = [r["error"] for r in runs if not finished(r)]
            if bad:
                log(f"solo {req['id']}: failed ({bad[0]}); not cached")
                continue
            ttfts = [r["first"] - r["send"] for r in runs]
            tpots = [tpot(r) for r in runs]
            cache["solo"][req["id"]] = {
                "prompt_tokens": req["prompt_tokens"],
                "ttft": statistics.median(ttfts), "tpot": _median(tpots),
                "output_tokens": runs[0]["output_tokens"],
                "finish_reason": runs[0]["finish_reason"],
                "runs": [[a, b, r["output_tokens"]]
                         for a, b, r in zip(ttfts, tpots, runs)]}
            write_json(path, cache)
            e = cache["solo"][req["id"]]
            log(f"solo {i + 1}/{len(todo)} {req['id']}: {req['prompt_tokens']} "
                f"tokens in, {e['output_tokens']} out, TTFT {e['ttft']:.3f} s, "
                f"TPOT {e['tpot'] * 1e3 if e['tpot'] else float('nan'):.1f} ms")
    return cache


def main():
    from .loadgen import add_client_args, client_kwargs
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--limit", type=int,
                    help="measure only the first N requests of the workload")
    ap.add_argument("--per-bucket", type=int,
                    help="measure only the first K requests of each "
                         "prompt-length bucket (for --solo-mode buckets)")
    ap.add_argument("--bucket-edges", type=int, nargs="+",
                    default=BUCKET_EDGES)
    add_client_args(ap)
    args = ap.parse_args()
    header, reqs = read_jsonl(workload_path(args.results, args.model,
                                            args.dataset))
    if args.limit:
        reqs = reqs[:args.limit]
    if args.per_bucket:
        reqs = select_for_buckets(reqs, args.per_bucket, args.bucket_edges)
    asyncio.run(profile([header["reference"]] + reqs, args.endpoint,
                        solo_path(args.results, args.model, args.dataset),
                        repeats=args.repeats, warmup=args.warmup,
                        **client_kwargs(args)))


if __name__ == "__main__":
    main()
