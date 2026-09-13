// The MonoFab kernel: one persistent kernel, one worker per SM, in which
// every tile is a device function.
#include <cstdio>

#include "fabric/core_tiles.cuh"
#include "fabric/tiles/attention.cuh"
#include "fabric/tiles/aux.cuh"
#include "fabric/tiles/gemm.cuh"
#include "fabric/launch.h"

namespace monofab {

struct CoreExecutor {
  struct Params {};
  __device__ __forceinline__ static void run(const TileDesc& t, const ProgramDesc* P, unsigned char* arena,
                             const Params& params, DeviceState& S) {
    (void)params;
    (void)S;
    if (run_core_tile(t, P)) return;
    switch (t.kind) {
      case kTileGemm: gemm::run(t, arena); break;
      case kTileAttnPrefill: attn::prefill(t, arena); break;
      case kTileAttnDecode: attn::decode(t, arena); break;
      case kTileAttnCombine: attn::combine(t); break;
      case kTileRmsNorm: aux::rmsnorm(t); break;
      case kTileQkRope: aux::qk_rope(t); break;
      case kTileRouter: aux::router(t); break;
      case kTileExpand: aux::expand(t, P, S); break;
      case kTilePermute: aux::permute(t); break;
      case kTileAddNorm: aux::add_norm(t); break;
      case kTileEmbed: aux::embed(t); break;
      case kTileSample: aux::sample(t); break;
      case kTileAttnPlan: aux::attn_plan(t, P); break;
      case kTileGemmAsym: gemm::run_asym(t, arena); break;
      case kTileAsymReduce: aux::asym_reduce(t); break;
      default:
        if (threadIdx.x == 0) printf("monofab: unknown tile kind %u\n", t.kind);
    }
  }
};

__global__ void __launch_bounds__(kWorkerThreads, 1)
    fabric_core_kernel(DeviceState* S, uint32_t first_worker) {
  extern __shared__ __align__(1024) unsigned char arena[];
  CoreExecutor::Params params;
  worker_loop<CoreExecutor>(S, params, arena, first_worker);
}

cudaError_t launch_fabric(DeviceState* S, int n_workers, size_t smem_bytes, cudaStream_t stream,
                          int first_worker) {
  cudaError_t err = cudaFuncSetAttribute(fabric_core_kernel,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem_bytes));
  if (err != cudaSuccess) return err;
  fabric_core_kernel<<<n_workers, kWorkerThreads, smem_bytes, stream>>>(
      S, static_cast<uint32_t>(first_worker));
  return cudaGetLastError();
}

size_t device_state_bytes() { return sizeof(DeviceState); }
size_t snapshot_offset(int i) { return offsetof(DeviceState, snap) + i * sizeof(Snapshot); }
size_t published_offset() { return offsetof(DeviceState, published); }
size_t shutdown_offset() { return offsetof(DeviceState, shutdown); }
size_t idle_cap_offset() { return offsetof(DeviceState, idle_cap); }
size_t trace_ptr_offset() { return offsetof(DeviceState, trace_ptr); }
size_t trace_cap_offset() { return offsetof(DeviceState, trace_cap); }
size_t trace_n_offset() { return offsetof(DeviceState, trace_n); }
size_t mirror_ptr_offset() { return offsetof(DeviceState, mirror); }
uint64_t wait_note_address() {
  void* p = nullptr;
  if (cudaGetSymbolAddress(&p, g_wait_note) != cudaSuccess) return 0;
  return reinterpret_cast<uint64_t>(p);
}
size_t worker_tile_offset() { return offsetof(DeviceState, worker_tile); }
size_t worker_tile_ns_offset() { return offsetof(DeviceState, worker_tile_ns); }
size_t lane_offset(int l) { return offsetof(DeviceState, lane) + l * sizeof(LaneState); }

LaneCounters lane_counter_offsets() {
  LaneCounters c;
  c.slots = offsetof(LaneState, slots);
  c.inflight = offsetof(LaneState, inflight);
  c.inflight_max = offsetof(LaneState, inflight_max);
  c.running = offsetof(LaneState, running);
  c.tag = offsetof(LaneState, tag);
  c.iter_in_prog = offsetof(LaneState, iter_in_prog);
  c.gen = offsetof(LaneState, gen);
  c.ring_head = offsetof(LaneState, ring_head);
  c.ring_reserve = offsetof(LaneState, ring_reserve);
  c.ring_tail = offsetof(LaneState, ring_tail);
  c.link_head = offsetof(LaneState, link_head);
  c.link_reserve = offsetof(LaneState, link_reserve);
  c.link_tail = offsetof(LaneState, link_tail);
  c.ring = offsetof(LaneState, ring);
  c.link = offsetof(LaneState, link);
  return c;
}

}  // namespace monofab
