# SPDX-License-Identifier: Apache-2.0
"""MuxWise's offline profile, producing its division table.

    python -m monoserve.baselines.muxwise.profile --model /path/to/model \\
        --profile profile.json --hot-fraction F --out divisions.yaml

The public MuxWise code reads a fixed table; this script builds one with
the offline method of the MuxWise paper. On the served model (the Local
baseline's, with the same hot tier), it measures
  - decode step latency alone, per decode batch size, on the decode share
    of every candidate division and on the whole GPU;
  - decode step latency beside a running prefill of each profiled size on
    the prefill share, per division and batch size;
and takes, per division and batch size, the worst slowdown over the
prefill sizes. The TPOT target is alpha times the decode step alone on the
whole GPU at batch size 1. For each batch size, the division with the
fewest decode SMs whose predicted step (alone times worst slowdown) meets
the target is chosen, and a division's threshold is the smallest batch
size that chooses it or a larger one; divisions never chosen are dropped,
and the table keeps MuxWise's format.
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def candidate_divisions(total_sms, step=8, least=16):
    """(prefill_sm, decode_sm) with decode shares from `least` up in
    granules of `step`, prefill shares in whole granules."""
    out = []
    d = least
    while total_sms - d >= least:
        p = (total_sms - d) // step * step
        if p >= least:
            out.append((p, total_sms - p))
        d += step
    return sorted(set(out), key=lambda x: x[1])


def measure(rt, batch_sizes, context, prefill_tokens, reps=8, alpha=5.0, partial=None):
    """Runs inside the executor (collective_rpc "muxwise_profile"): decode
    latencies alone and beside prefills on the groups of the runtime, with
    fake requests on free KV blocks. Per batch size, the divisions are
    walked from the largest decode share down, and the walk stops at the
    first one that misses the target, alone or beside a prefill: fewer
    decode SMs are never faster, so the smallest share that meets it is
    found without timing the slowest points. Each point is printed, and
    the measurements so far are saved to `partial` after every batch size."""
    import torch

    def note(msg):
        print(f"[muxwise profile] {msg}", flush=True)
    block = rt.block
    blocks_per_req = -(-(context + 1) // block)
    total_blocks = rt.runner.kv_cache_config.num_blocks
    max_bs = max(batch_sizes)
    need = max_bs * blocks_per_req + sum(-(-t // block) for t in prefill_tokens) + 1
    if need > total_blocks:
        raise ValueError(f"the profile needs {need} KV blocks, the cache has {total_blocks}")
    rng = np.random.default_rng(0)
    decode_blocks = [list(range(1 + i * blocks_per_req, 1 + (i + 1) * blocks_per_req))
                     for i in range(max_bs)]
    first_free = 1 + max_bs * blocks_per_req

    def decode_reqs(bs):
        return [(int(rng.integers(100, 1000)), context, decode_blocks[i]) for i in range(bs)]

    def prefill_reqs(tokens):
        blocks = list(range(first_free, first_free + -(-tokens // block)))
        return [("profile", rng.integers(100, 1000, size=tokens).tolist(), blocks)]

    def step(ds, reqs, temps):
        t0 = time.perf_counter()
        rt.decode(reqs, temps)
        ds.synchronize()
        return time.perf_counter() - t0

    def decode_time(group, bs, limit=None):
        """Mean decode step alone; a first step over `limit` ends the timing."""
        _, ds = rt.streams.groups[group]
        rt.cur = group
        reqs, temps = decode_reqs(bs), [0.0] * bs
        with torch.cuda.stream(ds):
            first = step(ds, reqs, temps)
            if limit is not None and first > limit:
                return first
            return float(np.mean([step(ds, reqs, temps) for _ in range(reps)]))

    def corun_time(group, bs, tokens, limit=None):
        """Mean decode step while one prefill of `tokens` runs on the
        prefill partition, over the steps that started before it ended; a
        step over `limit` ends the timing."""
        ps, ds = rt.streams.groups[group]
        rt.cur = group
        reqs, temps = decode_reqs(bs), [0.0] * bs
        with torch.cuda.stream(ps):
            batch = rt.prefill_begin(prefill_reqs(tokens), [0.0])
            rt.prefill_layers(batch, 0, rt.num_layers)
        times = []
        with torch.cuda.stream(ds):
            while not batch.done() and len(times) < 4096:
                times.append(step(ds, reqs, temps))
                if limit is not None and times[-1] > limit:
                    break
        ps.synchronize()
        return float(np.mean(times[:-1] if len(times) > 1 and times[-1] <= (limit or 1e30)
                             else times))

    last = len(rt.streams.groups) - 1
    out = {"sms": rt.streams.sms, "solo_full": {}, "solo": {}, "corun": {}}
    for bs in batch_sizes:
        out["solo_full"][bs] = decode_time(last, bs)
        note(f"decode alone on every SM, bs {bs}: {out['solo_full'][bs] * 1e3:.1f} ms")
    target = alpha * out["solo_full"][batch_sizes[0]]
    out["target"] = target
    note(f"TPOT target {target * 1e3:.1f} ms")
    for bs in batch_sizes:
        for g in sorted(range(1, last), key=lambda g: -rt.streams.sms[g][1]):
            solo = decode_time(g, bs, limit=target)
            out["solo"][f"{g}:{bs}"] = solo
            worst = solo
            if solo <= target:
                for t in prefill_tokens:
                    c = corun_time(g, bs, t, limit=target)
                    out["corun"][f"{g}:{bs}:{t}"] = c
                    worst = max(worst, c)
                    if c > target:
                        break
            note(f"bs {bs}, decode on {rt.streams.sms[g][1]} SMs: alone {solo * 1e3:.1f} ms, "
                 f"worst beside prefill {worst * 1e3:.1f} ms")
            if worst > target:
                break
        if partial:
            with open(partial, "w") as f:
                json.dump(out, f)
    rt.cur = 0
    torch.cuda.synchronize()
    return out


def table(data, batch_sizes, prefill_tokens, alpha):
    """MuxWise's rows [prefill_sm, decode_sm, decode_bs_threshold] from the
    measurements."""
    sms = [tuple(x) for x in data["sms"]]
    groups = range(1, len(sms) - 1)

    def get(d, key):   # keys come back as strings from JSON; unmeasured points are None
        return d.get(key, d.get(str(key)))

    target = alpha * get(data["solo_full"], batch_sizes[0])
    choice = {}
    for bs in batch_sizes:
        chosen = None
        for g in sorted(groups, key=lambda g: sms[g][1]):
            solo = get(data["solo"], f"{g}:{bs}")
            runs = [get(data["corun"], f"{g}:{bs}:{t}") for t in prefill_tokens]
            if solo is None or any(c is None for c in runs):
                continue   # not measured: the walk stopped above this share
            slow = max(c / solo for c in runs)   # the worst slowdown of the cell
            if solo * slow <= target:
                chosen = g
                break
        # none meets the target: the largest decode share
        choice[bs] = chosen if chosen is not None else max(groups, key=lambda g: sms[g][1])
    # thresholds grow with the decode share; a larger share starts right
    # after the largest batch size the smaller one was measured to serve
    rows, prev, prev_bs = [], None, 0
    for bs in batch_sizes:
        g = choice[bs]
        if prev is None or sms[g][1] > sms[prev][1]:
            rows.append([sms[g][0], sms[g][1], 1 if prev is None else prev_bs + 1])
            prev = g
        prev_bs = bs
    return rows, target


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--profile", default=None, help="activation profiles (python -m monoserve.profile)")
    ap.add_argument("--hot-fraction", type=float, default=0.0)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--out", required=True, help="division table to write (YAML)")
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--context", type=int, default=1024, help="decode context length profiled")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--prefill-tokens", default="1024,4096,16384")
    ap.add_argument("--division-step", type=int, default=8,
                    help="SMs between the decode shares of the candidate divisions")
    ap.add_argument("--reps", type=int, default=8, help="decode steps timed per point")
    ap.add_argument("--raw", default=None, help="also write the measurements (JSON)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()
    import torch
    from vllm import LLM

    from monoserve.baselines.muxwise.config import MuxConfig, dump
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    prefill_tokens = [int(x) for x in args.prefill_tokens.split(",")]
    total = torch.cuda.get_device_properties(0).multi_processor_count
    cands = candidate_divisions(total, step=args.division_step)
    probe = MuxConfig(sm_group_num=len(cands) + 2,
                      manual_divisions=[[p, d, 0] for p, d in cands])
    probe_path = args.out + ".candidates.json"
    with open(probe_path, "w") as f:
        json.dump({"sm_group_num": probe.sm_group_num,
                   "manual_divisions": probe.manual_divisions}, f)
    local = {"profile": os.path.abspath(args.profile) if args.profile else None,
             "hot_fraction": args.hot_fraction}
    llm = LLM(model=args.model, max_model_len=args.max_model_len, enforce_eager=True,
              enable_prefix_caching=False, async_scheduling=False,
              cpu_offload_gb=100000, cpu_offload_params={"w13_weight", "w2_weight"},
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_num_seqs=max(batch_sizes),
              distributed_executor_backend="monoserve.baselines.muxwise.executor.MuxWiseExecutor",
              scheduler_cls="monoserve.baselines.muxwise.scheduler.MuxWiseScheduler",
              additional_config={"local_moe": local, "muxwise": {"config": os.path.abspath(probe_path)}})
    partial = os.path.abspath(args.out + ".partial.json")
    data = llm.collective_rpc("muxwise_profile", args=(batch_sizes, args.context, prefill_tokens),
                              kwargs={"reps": args.reps, "alpha": args.alpha, "partial": partial})[0]
    if os.path.exists(partial):
        os.remove(partial)
    os.remove(probe_path)
    if args.raw:
        with open(args.raw, "w") as f:
            json.dump(data, f, indent=1)
    rows, target = table(data, batch_sizes, prefill_tokens, args.alpha)
    dump(MuxConfig(sm_group_num=len(rows) + 2, manual_divisions=rows), args.out)
    print(f"TPOT target {target * 1e3:.1f} ms; divisions {rows}; wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
