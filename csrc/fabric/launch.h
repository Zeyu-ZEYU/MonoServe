// Host-visible entry points of the fabric kernel translation unit.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace monofab {

struct DeviceState;

struct LaneCounters {
  size_t slots, inflight, inflight_max, running, tag, iter_in_prog, gen;
  size_t ring_head, ring_reserve, ring_tail, link_head, link_reserve, link_tail, ring, link;
};

// Workers first_worker .. first_worker + n_workers - 1, one per block.
cudaError_t launch_fabric(DeviceState* S, int n_workers, size_t smem_bytes, cudaStream_t stream,
                          int first_worker = 0);
size_t device_state_bytes();
size_t snapshot_offset(int i);
size_t published_offset();
size_t shutdown_offset();
size_t idle_cap_offset();
size_t trace_ptr_offset();
size_t trace_cap_offset();
size_t trace_n_offset();
size_t mirror_ptr_offset();
size_t worker_tile_offset();
size_t worker_tile_ns_offset();
// Device address of the long-wait records (wait_phase in runtime.cuh):
// uint64 [256 workers][12 warps][2].
uint64_t wait_note_address();
size_t lane_offset(int l);
LaneCounters lane_counter_offsets();

}  // namespace monofab
