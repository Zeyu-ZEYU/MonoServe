// Tiles the fabric itself needs: bookkeeping reset between iterations, the
// calibration probe, and the spin and read tiles used by the mechanism
// tests and microbenchmarks.
#pragma once
#include "fabric/runtime.cuh"

namespace monofab {

// Per-tile trace record written by tiles that carry a trace pointer.
struct alignas(32) TraceRec {
  unsigned long long start_ns;
  unsigned long long end_ns;
  unsigned int worker;
  unsigned int sm;
  unsigned int count;   // incremented once per execution
  unsigned int lane;    // lane the worker belonged to when it ran the tile
};

__device__ __forceinline__ void trace(unsigned long long base, uint32_t idx,
                                      unsigned long long t0, unsigned long long t1,
                                      uint32_t lane) {
  if (base == 0) return;
  TraceRec* r = reinterpret_cast<TraceRec*>(base) + idx;
  r->start_ns = t0;
  r->end_ns = t1;
  r->worker = worker_index();
  r->sm = smid();
  r->lane = lane;
  atomicAdd(&r->count, 1u);
}

// a[0]: duration in ns; a[1]: trace base; i0: trace index; i1: worker lane
__device__ void tile_spin(const TileDesc& t) {
  if (threadIdx.x == 0) {
    const unsigned long long t0 = globaltimer_ns();
    while (globaltimer_ns() - t0 < t.a[0]) {
    }
    trace(t.a[1], t.i0, t0, globaltimer_ns(), t.i1);
  }
}

// a[0]: source; a[1]: bytes; a[2]: sink (one word per worker);
// a[3]: trace base; i0: trace index. Every thread streams 16-byte loads.
__device__ void tile_read(const TileDesc& t) {
  __shared__ unsigned long long t0;
  if (threadIdx.x == 0) t0 = globaltimer_ns();
  const uint4* p = reinterpret_cast<const uint4*>(t.a[0]);
  const size_t n = t.a[1] / 16;
  unsigned int acc = 0;
  for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
    const uint4 v = __ldcs(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u && t.a[2] != 0)
    reinterpret_cast<unsigned int*>(t.a[2])[worker_index()] = acc;
  __syncthreads();
  if (threadIdx.x == 0) trace(t.a[3], t.i0, t0, globaltimer_ns(), t.i1);
}

// Dependent-load chase over a single-cycle permutation of 64-bit indices.
// a[0]: chain base; i0: loads; a[1]: result (ns per load, one u64 at i1).
__device__ void tile_probe(const TileDesc& t) {
  if (threadIdx.x == 0) {
    const volatile unsigned long long* base =
        reinterpret_cast<const volatile unsigned long long*>(t.a[0]);
    unsigned long long idx = 0;
    const unsigned long long t0 = globaltimer_ns();
    for (uint32_t i = 0; i < t.i0; ++i) idx = base[idx];
    const unsigned long long t1 = globaltimer_ns();
    unsigned long long* out = reinterpret_cast<unsigned long long*>(t.a[1]);
    out[t.i1] = (t1 - t0) * 1000ull / (t.i0 ? t.i0 : 1);   // picoseconds per load
    out[t.i1 + 1] = idx;
  }
}

// Re-arm the bookkeeping of stages [i0, i1) except stage i2.
__device__ void tile_zero_runtime(const TileDesc& t, const ProgramDesc* P) {
  for (uint32_t s = t.i0 + threadIdx.x; s < t.i1; s += blockDim.x) {
    if (s == t.i2) continue;
    P->runtime[s].done = 0;
    P->runtime[s].deps = P->stages[s].deps_init;
  }
  __threadfence();
}

__device__ __forceinline__ bool run_core_tile(const TileDesc& t, const ProgramDesc* P) {
  switch (t.kind) {
    case kTileNop: return true;
    case kTileSpin: tile_spin(t); return true;
    case kTileRead: tile_read(t); return true;
    case kTileProbe: tile_probe(t); return true;
    case kTileZeroRuntime: tile_zero_runtime(t, P); return true;
    default: return false;
  }
}

}  // namespace monofab
