"""Disjoint SM partitions for the co-run experiments.

torch.cuda.GreenContext.create() always splits from the full device, so
two partitions created that way overlap on the same SMs. This module
carves disjoint groups itself through the CUDA driver API: successive
cuDevSmResourceSplitByCount calls split the running remainder, and each
group gets its own green context and stream.

    import partitions
    mod = partitions.build()
    sizes = partitions.create(mod, [64, 64])     # actual SM counts
    with partitions.on_partition(mod, 0) as s:   # torch ExternalStream
        with torch.cuda.stream(s):
            ...
    mod.partitions_destroy()

Running this file launches a spin kernel on two concurrent partitions and
checks that the SM ids they report are disjoint.
"""
import contextlib

import torch
from torch.utils.cpp_extension import load_inline

CU_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <vector>

#define DRV(call)                                                          \
  do {                                                                     \
    CUresult _st = (call);                                                 \
    if (_st != CUDA_SUCCESS) {                                             \
      const char* msg = nullptr;                                           \
      cuGetErrorString(_st, &msg);                                         \
      TORCH_CHECK(false, "driver call failed: ", #call, " -> ",            \
                  msg ? msg : "?");                                        \
    }                                                                      \
  } while (0)

namespace {
struct Partition {
  CUgreenCtx gctx;
  CUcontext ctx;
  CUstream stream;
  int sm_count;
};
std::vector<Partition> g_parts;
std::vector<CUgreenCtx> g_scaffolds;  // keep the remainder contexts alive
}  // namespace

std::vector<int64_t> partitions_create(std::vector<int64_t> counts) {
  TORCH_CHECK(g_parts.empty(), "partitions already created");
  CUdevice dev;
  DRV(cuDeviceGet(&dev, 0));
  CUdevResource input{};
  DRV(cuDeviceGetDevResource(dev, &input, CU_DEV_RESOURCE_TYPE_SM));
  std::vector<int64_t> actual;
  for (size_t k = 0; k < counts.size(); ++k) {
    CUdevResource group{};
    CUdevResource remaining{};
    unsigned int nb = 1;
    DRV(cuDevSmResourceSplitByCount(&group, &nb, &input, &remaining, 0,
                                    (unsigned int)counts[k]));
    TORCH_CHECK(nb == 1, "split produced ", nb, " groups");
    CUdevResourceDesc desc;
    DRV(cuDevResourceGenerateDesc(&desc, &group, 1));
    Partition p{};
    DRV(cuGreenCtxCreate(&p.gctx, desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
    DRV(cuCtxFromGreenCtx(&p.ctx, p.gctx));
    DRV(cuGreenCtxStreamCreate(&p.stream, p.gctx, CU_STREAM_NON_BLOCKING,
                               0));
    p.sm_count = (int)group.sm.smCount;
    g_parts.push_back(p);
    actual.push_back(p.sm_count);
    if (k + 1 < counts.size()) {
      // A split accepts only resources queried from a device or a
      // context, not a previous split's remainder: materialize the
      // remainder as a scaffold green context and query it.
      CUdevResourceDesc rem_desc;
      DRV(cuDevResourceGenerateDesc(&rem_desc, &remaining, 1));
      CUgreenCtx scaffold;
      DRV(cuGreenCtxCreate(&scaffold, rem_desc, dev,
                           CU_GREEN_CTX_DEFAULT_STREAM));
      g_scaffolds.push_back(scaffold);
      DRV(cuGreenCtxGetDevResource(scaffold, &input,
                                   CU_DEV_RESOURCE_TYPE_SM));
    }
  }
  return actual;
}

void partition_push(int64_t i) {
  TORCH_CHECK(i >= 0 && (size_t)i < g_parts.size(), "bad partition index");
  DRV(cuCtxPushCurrent(g_parts[i].ctx));
}

void partition_pop() {
  CUcontext dummy;
  DRV(cuCtxPopCurrent(&dummy));
}

int64_t partition_stream(int64_t i) {
  TORCH_CHECK(i >= 0 && (size_t)i < g_parts.size(), "bad partition index");
  return (int64_t)(uintptr_t)g_parts[i].stream;
}

void partitions_destroy() {
  for (auto& p : g_parts) {
    cuStreamDestroy(p.stream);
    cuGreenCtxDestroy(p.gctx);
  }
  g_parts.clear();
  for (auto& s : g_scaffolds) cuGreenCtxDestroy(s);
  g_scaffolds.clear();
}

__global__ void smid_spin(int* out, long long cycles) {
  if (threadIdx.x == 0) {
    unsigned smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    out[blockIdx.x] = (int)smid;
  }
  long long start = clock64();
  while (clock64() - start < cycles) {
  }
}

void launch_smid_spin(torch::Tensor out, int64_t cycles, int64_t blocks,
                      int64_t threads) {
  smid_spin<<<(int)blocks, (int)threads, 0,
              at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int>(),
                                                  (long long)cycles);
}
"""

CPP_SRC = r"""
#include <vector>
std::vector<int64_t> partitions_create(std::vector<int64_t> counts);
void partitions_destroy();
void partition_push(int64_t i);
void partition_pop();
int64_t partition_stream(int64_t i);
void launch_smid_spin(torch::Tensor out, int64_t cycles, int64_t blocks,
                      int64_t threads);
"""

FUNCS = ["partitions_create", "partitions_destroy", "partition_push",
         "partition_pop", "partition_stream", "launch_smid_spin"]


def build():
    return load_inline(name="monoserve_smpart", cpp_sources=CPP_SRC,
                       cuda_sources=CU_SRC, functions=FUNCS, with_cuda=True,
                       verbose=False, extra_cuda_cflags=["-O2"],
                       extra_ldflags=["-lcuda"])


def create(mod, counts):
    """Carve disjoint partitions of the requested SM counts. The driver
    split needs a current context in this thread, which the first runtime
    allocation creates."""
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    return list(mod.partitions_create(list(counts)))


@contextlib.contextmanager
def on_partition(mod, i):
    """Make partition i's context current and yield its stream as a torch
    ExternalStream. Launch work under torch.cuda.stream(yielded)."""
    mod.partition_push(i)
    try:
        yield torch.cuda.ExternalStream(mod.partition_stream(i))
    finally:
        mod.partition_pop()


def _audit():
    mod = build()
    total = torch.cuda.get_device_properties(0).multi_processor_count
    sizes = create(mod, [total // 2, total // 4])
    print(f"device SMs: {total}; partitions: {sizes}")
    outs = [torch.full((4 * n,), -1, dtype=torch.int32, device="cuda")
            for n in sizes]
    torch.cuda.synchronize()
    streams = []
    for i, n in enumerate(sizes):
        with on_partition(mod, i) as s:
            with torch.cuda.stream(s):
                mod.launch_smid_spin(outs[i], int(0.2 * 1.98e9), 4 * n, 128)
            streams.append(s)
    for s in streams:
        s.synchronize()
    seen = [set(o.tolist()) - {-1} for o in outs]
    for i, s in enumerate(seen):
        print(f"partition {i}: {len(s)} distinct SMs")
    print("DISJOINT" if not (seen[0] & seen[1]) else "OVERLAP")
    mod.partitions_destroy()


if __name__ == "__main__":
    _audit()
