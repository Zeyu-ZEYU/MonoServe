"""Serve a model with MonoServe behind vLLM's OpenAI-compatible API server.

    python -m monoserve.serve --model /path/to/Qwen3-30B-A3B-Instruct-2507 \\
        --calibration calibration.json [--profile profile.json] [--kv-gb 38] \\
        [--port 8000] [-- extra vllm serve arguments]

The options below go to the engine as vLLM's additional config; the rest of
the command line goes to `vllm serve` unchanged.
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--kv-gb", type=float, default=None)
    ap.add_argument("--hot-fraction", type=float, default=None)
    ap.add_argument("--hbm-budget-gb", type=float, default=None)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--prefill-lanes", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--forms", default="16,32,64,128,0",
                    help="kernel forms for DRAM experts: tile heights, 0 for asymmetric")
    ap.add_argument("--no-contention-awareness", action="store_true",
                    help="plan without gamma and run without the credit gate (ablation)")
    ap.add_argument("--no-staging", action="store_true", help="no staging buffers (ablation)")
    ap.add_argument("--engine", choices=("full", "ml", "kf"), default="full",
                    help="ablations: ml runs one mixed batch at a time on every SM, kf runs the "
                         "lanes on green contexts instead of the kernel fabric")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    opts = {"calibration": os.path.abspath(args.calibration),
            "profile": os.path.abspath(args.profile) if args.profile else None,
            "kv_gb": args.kv_gb, "hot_fraction": args.hot_fraction,
            "hbm_budget_gb": args.hbm_budget_gb, "token_budget": args.token_budget,
            "prefill_lanes": args.prefill_lanes, "alpha": args.alpha,
            "forms": [int(f) for f in args.forms.split(",")],
            "contention_aware": not args.no_contention_awareness,
            "staging": not args.no_staging, "engine": args.engine}
    # vLLM imports the executor and scheduler by name, in this interpreter
    # and in its engine process; without an install, the package must be on
    # their path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = root + (os.pathsep + path if path else "")
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
           "--distributed-executor-backend", "monoserve.vllm.executor.MonoServeExecutor",
           "--scheduler-cls", "monoserve.vllm.scheduler.MonoServeScheduler",
           "--no-async-scheduling", "--block-size", "64", "--no-enable-prefix-caching",
           "--generation-config", "vllm", "--max-model-len", str(args.max_model_len),
           "--additional-config", json.dumps({"monoserve": opts}),
           "--host", args.host, "--port", str(args.port)] + rest
    print(" ".join(cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
