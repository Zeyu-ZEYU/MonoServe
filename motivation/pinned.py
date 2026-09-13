"""Pinned host memory mapped into the GPU address space.

Expert weights that stay in CPU DRAM are read by GPU kernels through a
device pointer to page-locked host memory, with no copy into HBM. On a
GH200 these reads cross NVLink-C2C; on a PCIe-attached GPU they cross
PCIe. The returned CUDA tensor aliases the pinned buffer, so the caller
must keep the pinned tensor alive.
"""
import torch
from torch.utils.cpp_extension import load_inline

_ext = None

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

torch::Tensor wrap_pinned_as_cuda(torch::Tensor pinned) {
  TORCH_CHECK(pinned.is_pinned(), "tensor must be pinned host memory");
  void* dptr = nullptr;
  cudaError_t err = cudaHostGetDevicePointer(&dptr, pinned.data_ptr(), 0);
  TORCH_CHECK(err == cudaSuccess, "cudaHostGetDevicePointer failed: ",
              cudaGetErrorString(err));
  auto opts = torch::TensorOptions()
                  .dtype(pinned.dtype())
                  .device(torch::kCUDA, at::cuda::current_device());
  return torch::from_blob(dptr, pinned.sizes(), pinned.strides(), opts);
}
"""


def _module():
    global _ext
    if _ext is None:
        _ext = load_inline(name="monoserve_pinned_view", cpp_sources=_SRC,
                           functions=["wrap_pinned_as_cuda"], with_cuda=True,
                           verbose=False)
    return _ext


def to_pinned_cuda_view(t):
    """Copy t into pinned host memory. Returns (pinned, cuda_view)."""
    pinned = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    pinned.copy_(t)
    return pinned, _module().wrap_pinned_as_cuda(pinned)
