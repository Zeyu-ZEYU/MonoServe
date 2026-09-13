"""Serve a model with the EP baseline: stock vLLM sharding the experts
across the GPUs of one node with expert parallelism, attention data
parallel, and an all-to-all exchange at every MoE layer. vLLM's
data-parallel API server spreads requests over the ranks.

    python -m monoserve.baselines.serve_ep --model /path/to/model --gpus 8 \\
        [--port 8000] [-- extra vllm serve arguments]
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
           "--data-parallel-size", str(args.gpus),
           "--enable-expert-parallel", "--generation-config", "vllm",
           "--max-model-len", str(args.max_model_len), "--host", args.host,
           "--port", str(args.port)] + rest
    print(" ".join(cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
