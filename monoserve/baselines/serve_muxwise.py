"""Serve a model with the MuxWise baseline: prefill-decode multiplexing on
green contexts, on the Local baseline's model and expert placement.

    python -m monoserve.baselines.serve_muxwise --model /path/to/model \\
        --divisions divisions.yaml --profile profile.json --hot-fraction F \\
        [--port 8000] [-- extra vllm serve arguments]

divisions.yaml is MuxWise's division table, from
python -m monoserve.baselines.muxwise.profile.
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--divisions", required=True, help="MuxWise's division table (YAML)")
    ap.add_argument("--profile", default=None, help="activation profiles (python -m monoserve.profile)")
    ap.add_argument("--hot-fraction", type=float, default=0.0)
    ap.add_argument("--hot-experts", type=int, default=None)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    local = {"profile": os.path.abspath(args.profile) if args.profile else None,
             "hot_fraction": args.hot_fraction}
    if args.hot_experts is not None:
        local["hot_experts"] = args.hot_experts
    extra = {"local_moe": local, "muxwise": {"config": os.path.abspath(args.divisions)}}
    # the baselines' modules must import in vLLM's processes, installed or not
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = root + (os.pathsep + path if path else "")
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
           "--cpu-offload-gb", "100000", "--cpu-offload-params", "w13_weight", "w2_weight",
           "--distributed-executor-backend", "monoserve.baselines.muxwise.executor.MuxWiseExecutor",
           "--scheduler-cls", "monoserve.baselines.muxwise.scheduler.MuxWiseScheduler",
           "--no-async-scheduling", "--enforce-eager", "--no-enable-prefix-caching",
           "--additional-config", json.dumps(extra),
           "--generation-config", "vllm", "--max-model-len", str(args.max_model_len),
           "--host", args.host, "--port", str(args.port)] + rest
    print(" ".join(cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
