"""Load-to-use latency of a cache-missing read from HBM and from CPU DRAM.

One CUDA thread walks a single-cycle random permutation (Sattolo) laid out
at a 128-byte node stride. Every load depends on the previous one, so the
elapsed time divided by the number of loads is the round trip of one miss:
from the L2 to HBM, or over the link to CPU DRAM. Footprints far above the
L2 capacity make almost every load miss; a 16 MB footprint that fits in
the L2 is kept as a control for the probe itself. CUDA events bracket the
kernel, so no SM clock assumption enters the number.

Section 2.3 reports about 340 ns for an HBM miss and 830 ns for a miss to
CPU DRAM over NVLink-C2C on a GH200.

    python latency_probe.py [--iters 1000000] [--footprints-mb 16 96 192 384]
"""
import argparse

import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

from pinned import to_pinned_cuda_view

STRIDE_U64 = 16   # 16 x 8 B = 128 B between nodes


def _module():
    return load_inline(
        name="monoserve_latency_chase",
        cpp_sources=r"""
#include <torch/extension.h>
void chase_launch(torch::Tensor buf, long long iters, torch::Tensor out);
""",
        cuda_sources=r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void chase_kernel(const unsigned long long* __restrict__ buf,
                             long long iters, unsigned long long* out) {
  unsigned long long idx = *out;  // continue where the last launch stopped
  for (long long i = 0; i < iters; ++i) idx = buf[idx];
  *out = idx;
}

void chase_launch(torch::Tensor buf, long long iters, torch::Tensor out) {
  chase_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const unsigned long long*>(buf.data_ptr()), iters,
      reinterpret_cast<unsigned long long*>(out.data_ptr()));
}
""",
        functions=["chase_launch"], with_cuda=True, verbose=False)


def sattolo_cycle(n_nodes, seed):
    """buf[i * STRIDE] holds the index of the next node's slot."""
    rng = np.random.default_rng(seed)
    perm = np.arange(n_nodes)
    for i in range(n_nodes - 1, 0, -1):
        j = rng.integers(0, i)
        perm[i], perm[j] = perm[j], perm[i]
    nxt = np.empty(n_nodes, dtype=np.uint64)
    nxt[perm[:-1]] = perm[1:]
    nxt[perm[-1]] = perm[0]
    buf = np.zeros(n_nodes * STRIDE_U64, dtype=np.uint64)
    buf[::STRIDE_U64] = nxt * STRIDE_U64
    return buf


def per_load_ns(mod, buf, iters, reps=3):
    out = torch.zeros(1, dtype=torch.int64, device="cuda")
    mod.chase_launch(buf, 20_000, out)   # warm the TLB and the module
    torch.cuda.synchronize()
    best = None
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        mod.chase_launch(buf, iters, out)
        e1.record()
        torch.cuda.synchronize()
        ns = e0.elapsed_time(e1) * 1e6 / iters
        best = ns if best is None else min(best, ns)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1_000_000)
    ap.add_argument("--footprints-mb", type=int, nargs="+",
                    default=[16, 96, 192, 384])
    args = ap.parse_args()
    torch.cuda.init()
    mod = _module()
    print(f"{'footprint':>10s} {'HBM (ns)':>10s} {'CPU DRAM (ns)':>14s}")
    for mb in args.footprints_mb:
        host = torch.from_numpy(
            sattolo_cycle(mb * 2**20 // (STRIDE_U64 * 8), seed=7).view(np.int64))
        hbm = host.cuda()
        t_hbm = per_load_ns(mod, hbm, args.iters)
        del hbm
        torch.cuda.empty_cache()
        keep, view = to_pinned_cuda_view(host)
        t_cpu = per_load_ns(mod, view, args.iters)
        del keep, view
        note = "  (HBM column: L2-resident control)" if mb <= 32 else ""
        print(f"{mb:8d}MB {t_hbm:10.1f} {t_cpu:14.1f}{note}", flush=True)


if __name__ == "__main__":
    main()
