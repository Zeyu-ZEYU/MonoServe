"""Latency metrics and SLO scoring of harness records.

TTFT is first - send. TPOT is (t_last - t_first) / (n_out - 1): t_first
and t_last are the times of the first and the last chunk that carry tokens,
and n_out is the request's output_tokens. A request with fewer than two
output tokens has no TPOT; it meets the TPOT side of its SLO trivially and
stays out of the TPOT percentile. A request that did not finish (an error,
or still running when the run ended) violates its SLO, and its TTFT and TPOT
count as infinite in the percentiles, so a P99 becomes infinite once more
than 1% of the requests did not finish. Percentiles use the nearest-rank
definition. Goodput is the rate sustained at 90% attainment (goodput()).

    python -m eval.metrics --model qwen3-30b --dataset sharegpt [--alpha 5]
"""
import argparse
import json
import math

from .common import find_runs, read_jsonl, solo_path, workload_path

INF = math.inf


def finished(r):
    return r["error"] is None and r["finish"] is not None


def ttft(r):
    return None if r["first"] is None else r["first"] - r["send"]


def tpot(r):
    n = r["output_tokens"]
    if not finished(r) or not r["chunk_t"] or n is None or n < 2:
        return None
    return (r["chunk_t"][-1] - r["chunk_t"][0]) / (n - 1)


def percentile(vals, q):
    """Nearest-rank percentile; None for no values."""
    if not vals:
        return None
    s = sorted(vals)
    return s[max(0, math.ceil(q / 100 * len(s)) - 1)]


def slo_met(r, target):
    """Both TTFT and TPOT within target (a solo.Target)."""
    if not finished(r) or r["first"] is None:
        return False
    b = tpot(r)
    return ttft(r) <= target.ttft and (b is None or target.tpot is None
                                       or b <= target.tpot)


def summarize(records, targets):
    """Attainment and tails of one run. targets maps request id to a
    solo.Target; the TTFT ratio divides by the solo TTFT that sets the
    request's target (target.ttft_ref)."""
    met, ratios, tpots = 0, [], []
    for r in records:
        t = targets[r["id"]]
        if not finished(r) or r["first"] is None:
            ratios.append(INF)
            tpots.append(INF)
            continue
        met += slo_met(r, t)
        ratios.append(ttft(r) / t.ttft_ref)
        b = tpot(r)
        if b is not None:
            tpots.append(b)
    n = len(records)
    p99_tpot = percentile(tpots, 99)
    return {
        "n": n,
        "finished": sum(map(finished, records)),
        "attainment": met / n if n else None,
        "p99_ttft_ratio": percentile(ratios, 99),
        "p99_tpot_ms": None if p99_tpot is None else p99_tpot * 1e3,
        "max_send_lag_s": max((r["send"] - r["arrival"] for r in records
                               if r["send"] is not None), default=None),
    }


def goodput(rates, attainment, level=0.9):
    """Rate sustained at `level` attainment. Walking up the sweep, the first
    point below the level and the point before it are joined by a straight
    line, and the rate where the line crosses the level is returned. None if
    the lowest rate is already below the level. If no point falls below it,
    the highest swept rate is returned, which is then only a lower bound
    (see censored())."""
    prev = None
    for r, a in sorted(zip(rates, attainment)):
        if a < level:
            if prev is None:
                return None
            r0, a0 = prev
            return r0 + (a0 - level) / (a0 - a) * (r - r0)
        prev = (r, a)
    return None if prev is None else prev[0]


def censored(attainment, level=0.9):
    """True when the sweep never falls below the level."""
    return bool(attainment) and min(attainment) >= level


def sweep(results, model, dataset, alpha=5.0, solo_mode="exact",
          methods=None, level=0.9):
    """Score every run of one model and dataset. Returns
    {method: {"rates", "points" (summaries), "goodput", "censored",
    "headers", "targets"}}, methods in the order given or sorted."""
    from .solo import load_solo, targets as make_targets
    header, reqs = read_jsonl(workload_path(results, model, dataset))
    tg = make_targets(reqs, load_solo(solo_path(results, model, dataset)),
                      alpha, mode=solo_mode)
    runs = find_runs(results, model, dataset)
    out = {}
    for m in (methods or sorted(runs)):
        if m not in runs:
            continue
        rates, points, headers = [], [], []
        for rate, path, h in runs[m]:
            _, recs = read_jsonl(path)
            rates.append(rate)
            points.append(summarize(recs, tg))
            headers.append(h)
        att = [p["attainment"] for p in points]
        out[m] = {"rates": rates, "points": points, "headers": headers,
                  "goodput": goodput(rates, att, level),
                  "censored": censored(att, level), "targets": tg}
    return out


def fmt(v, spec):
    if v is None:
        return "-"
    return "inf" if v == INF else format(v, spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--solo-mode", choices=("exact", "buckets"),
                    default="exact")
    ap.add_argument("--level", type=float, default=0.9)
    ap.add_argument("--json", help="also write the summaries here")
    args = ap.parse_args()
    res = sweep(args.results, args.model, args.dataset, args.alpha,
                args.solo_mode, args.methods, args.level)
    for m, s in res.items():
        print(f"\n{m}  ({args.model}, {args.dataset}, alpha={args.alpha:g})")
        print(f"{'rate':>8} {'n':>6} {'done':>6} {'SLO %':>7} "
              f"{'P99 TTFT/solo':>14} {'P99 TPOT ms':>12} {'send lag s':>11}")
        for rate, p in zip(s["rates"], s["points"]):
            print(f"{rate:>8g} {p['n']:>6} {p['finished']:>6} "
                  f"{fmt(p['attainment'] and p['attainment'] * 100, '7.1f'):>7} "
                  f"{fmt(p['p99_ttft_ratio'], '14.2f'):>14} "
                  f"{fmt(p['p99_tpot_ms'], '12.1f'):>12} "
                  f"{fmt(p['max_send_lag_s'], '11.3f'):>11}")
        g = s["goodput"]
        bound = ">= " if s["censored"] else ""
        print(f"rate at {args.level:.0%} attainment: "
              f"{'never reached' if g is None else f'{bound}{g:.3g} req/s'}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({m: {k: v for k, v in s.items() if k != "targets"}
                       for m, s in res.items()}, f, indent=1, default=str)


if __name__ == "__main__":
    main()
