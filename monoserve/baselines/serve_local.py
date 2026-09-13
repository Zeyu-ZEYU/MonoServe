"""Serve a model with the Local baseline: stock vLLM with the expert set in
CPU DRAM and MonoServe's hot tier.

    python -m monoserve.baselines.serve_local --model /path/to/model \\
        --profile profile.json --hot-fraction 0.82 [--port 8000] \\
        [-- extra vllm serve arguments]

Expert weights (not their scales) go to pinned CPU DRAM through vLLM's UVA
offloading; the plugin in local_moe.py then copies the hot tier into HBM.
Everything else is vLLM's default configuration.
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--profile", default=None, help="activation profiles (python -m monoserve.profile)")
    ap.add_argument("--hot-fraction", type=float, default=0.0,
                    help="share of the MoE layers' experts kept in HBM")
    ap.add_argument("--hot-experts", type=int, default=None,
                    help="number of experts kept in HBM (instead of --hot-fraction)")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    conf = {"profile": os.path.abspath(args.profile) if args.profile else None,
            "hot_fraction": args.hot_fraction}
    if args.hot_experts is not None:
        conf["hot_experts"] = args.hot_experts
    # the plugin must import in vLLM's processes, installed or not
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = root + (os.pathsep + path if path else "")
    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", args.model,
           # every expert weight in pinned CPU DRAM, read over the link
           "--cpu-offload-gb", "100000", "--cpu-offload-params", "w13_weight", "w2_weight",
           "--additional-config", json.dumps({"local_moe": conf}),
           "--generation-config", "vllm", "--max-model-len", str(args.max_model_len),
           "--host", args.host, "--port", str(args.port)] + rest
    print(" ".join(cmd), flush=True)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    sys.exit(main())
