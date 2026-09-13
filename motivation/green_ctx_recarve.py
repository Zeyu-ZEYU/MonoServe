"""Cost of re-carving green-context SM partitions (Section 3.2).

A driver-level partition change destroys the current green contexts and
carves new ones. This script times full carve and destroy cycles of a
two-partition split, then the first kernel launch into a freshly carved
context, which also pays for loading the module into that context.
Section 3.2 reports 7.58 ms per re-carve on a GH200.

    python green_ctx_recarve.py [--split 64 32] [--reps 5]
"""
import argparse
import statistics
import time

import torch

import partitions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=int, nargs=2, default=[64, 32])
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    mod = partitions.build()
    torch.zeros(1, device="cuda")

    carve = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        partitions.create(mod, args.split)
        carve.append((time.perf_counter() - t0) * 1e3)
        mod.partitions_destroy()
    print(f"carve {args.split} (contexts + streams): "
          f"median {statistics.median(carve):.2f} ms, "
          f"samples {['%.2f' % c for c in carve]}")

    first, warm = [], []
    out = torch.zeros(64, dtype=torch.int32, device="cuda")
    for _ in range(args.reps):
        partitions.create(mod, args.split)
        with partitions.on_partition(mod, 0) as s:
            with torch.cuda.stream(s):
                t0 = time.perf_counter()
                mod.launch_smid_spin(out, 1000, 8, 128)
                s.synchronize()
                first.append((time.perf_counter() - t0) * 1e3)
                t0 = time.perf_counter()
                mod.launch_smid_spin(out, 1000, 8, 128)
                s.synchronize()
                warm.append((time.perf_counter() - t0) * 1e3)
        mod.partitions_destroy()
    print(f"first launch into a fresh context: median "
          f"{statistics.median(first):.2f} ms; later launches: median "
          f"{statistics.median(warm):.3f} ms")


if __name__ == "__main__":
    main()
