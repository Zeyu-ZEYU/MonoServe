"""MonoServe: MoE serving on CPU-GPU superchips with experts in CPU DRAM,
a contention-aware control plane, and a contention-gated kernel fabric."""

import torch  # noqa: F401  (loads the libraries the native extension links against)

__version__ = "0.1.0"
