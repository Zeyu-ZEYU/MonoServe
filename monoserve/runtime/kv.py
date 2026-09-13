"""Paged KV cache: per layer, K and V pages of 64 tokens,
[pages][64][kv_heads][128] bf16, with TMA maps for the attention tiles."""
import torch

from monoserve.runtime.kinds import HEAD_DIM, PAGE


def kv_map(fab, cache):
    pages, page, hkv, d = cache.shape
    return fab.tensor_map(cache.data_ptr(), [64, page, 2, hkv, pages],
                          [hkv * d * 2, 128, d * 2, page * hkv * d * 2], [64, 64, 2, 1, 1])


class KVCache:
    def __init__(self, fab, cfg, num_pages):
        self.num_pages = num_pages
        shape = (num_pages, PAGE, cfg.num_kv_heads, HEAD_DIM)
        self.k = [torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
                  for _ in range(cfg.num_layers)]
        self.v = [torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
                  for _ in range(cfg.num_layers)]
        self.k_maps = [kv_map(fab, t) for t in self.k]
        self.v_maps = [kv_map(fab, t) for t in self.v]
        self.free = list(range(num_pages))

    def allocate(self, n):
        if n > len(self.free):
            raise MemoryError("out of KV-cache pages")
        pages, self.free = self.free[:n], self.free[n:]
        return pages

    def release(self, pages):
        self.free.extend(pages)
