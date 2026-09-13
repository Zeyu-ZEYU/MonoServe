"""Request-rate sweep of one method against its endpoints.

At every rate, the load generator sends the workload's first
--num-requests requests (or every arrival within --duration seconds) as a
Poisson stream at that cluster-wide rate, routing each request to the
endpoint with the fewest outstanding requests. Each run is saved to
<results>/<model>/<dataset>/<method>/rate_<r>.jsonl: a header line and one
record per request (see loadgen.py). A rate whose file already exists is
skipped unless --overwrite is given. The sweep pauses --pause seconds
between rates so that the servers are idle when the next rate starts.

    python -m eval.run_sweep --model qwen3-30b --dataset sharegpt \
        --method MonoServe --rates 4 8 16 32 64 --num-requests 1000 \
        --endpoints http://<server-1>:8000 http://<server-2>:8000

--meta key=value (repeatable) stores extra settings in the header, for
example --meta hot_share=41.2 for the hot-tier sensitivity figure.

Every request carries its SLO targets (alpha times its solo latency, as
eval.metrics scores it) as vLLM extra args, which MonoServe's admission
plans for and the other methods ignore; --no-send-targets leaves them out.
"""
import argparse
import asyncio
import datetime
import json
import time

from .common import log, read_jsonl, run_path, solo_path, workload_path, write_jsonl
from .loadgen import add_client_args, client_kwargs, run
from .solo import load_solo, targets


def meta_value(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", required=True,
                    help="model name in the results layout")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True,
                    help="method name, stored in the run headers")
    ap.add_argument("--endpoints", nargs="+", required=True,
                    help="base URLs of the servers, one per GPU server")
    ap.add_argument("--rates", type=float, nargs="+", required=True,
                    help="cluster-wide request rates (req/s)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--num-requests", type=int)
    group.add_argument("--duration", type=float,
                       help="seconds of arrivals per rate")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed of the arrival pattern")
    ap.add_argument("--alpha", type=float, default=5.0,
                    help="SLO scale of the targets sent with the requests")
    ap.add_argument("--no-send-targets", action="store_true",
                    help="send no SLO targets; a run then serves any alpha "
                         "the metrics score it with")
    ap.add_argument("--solo-mode", choices=("exact", "buckets"), default="exact",
                    help="solo latency per request, or interpolated over "
                         "prompt-length buckets (see eval.solo --per-bucket)")
    ap.add_argument("--drain-timeout", type=float, default=600.0,
                    help="seconds after the last arrival before unfinished "
                         "requests are cancelled")
    ap.add_argument("--pause", type=float, default=10.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--meta", nargs="+", default=[], metavar="KEY=VALUE")
    add_client_args(ap)
    args = ap.parse_args()

    wl_header, reqs = read_jsonl(workload_path(args.results, args.model,
                                               args.dataset))
    meta = {k: meta_value(v) for k, v in (m.split("=", 1) for m in args.meta)}
    kw = client_kwargs(args)
    sent = None
    if not args.no_send_targets:
        solo = load_solo(solo_path(args.results, args.model, args.dataset))
        sent = {rid: (t.ttft, t.tpot) for rid, t in
                targets(reqs, solo, args.alpha, mode=args.solo_mode).items()}
    for i, rate in enumerate(args.rates):
        out = run_path(args.results, args.model, args.dataset, args.method,
                       rate)
        if out.exists() and not args.overwrite:
            log(f"skip rate {rate:g}: {out} exists")
            continue
        if i and args.pause:
            time.sleep(args.pause)
        log(f"{args.method} {args.model} {args.dataset}: rate {rate:g} req/s")
        started = datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds")
        records, info = asyncio.run(run(
            reqs, args.endpoints, rate, num=args.num_requests,
            duration=args.duration, seed=args.seed,
            drain_timeout=args.drain_timeout, targets=sent, **kw))
        header = {
            "method": args.method, "model": args.model,
            "dataset": args.dataset, "rate": rate,
            "num_requests": len(records), "duration": args.duration,
            "seed": args.seed, "alpha": args.alpha,
            "targets_sent": sent is not None, "solo_mode": args.solo_mode,
            "endpoints": args.endpoints,
            "served_model": info["served_model"], "sampling": kw["sampling"],
            "continuous_usage": args.continuous_usage,
            "drain_timeout": args.drain_timeout, "started": started,
            "wall_s": info["wall_s"], "meta": meta,
            "workload": {k: v for k, v in wl_header.items()
                         if k != "reference"},
        }
        write_jsonl(out, header, records)
        bad = sum(r["error"] is not None for r in records)
        log(f"wrote {out}: {len(records)} requests, {bad} not finished, "
            f"{info['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
