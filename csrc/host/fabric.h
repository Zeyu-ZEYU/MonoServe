// Host side of MonoFab: owns the device state, uploads lane programs,
// publishes plans as epochs, and reads the progress the kernel mirrors
// into pinned host memory.
#pragma once
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "fabric/types.h"

namespace monofab {

struct DeviceState;

// Destroys the streams and frees the device and pinned host memory now or,
// while a fabric kernel runs on the device, once the last one stops:
// cudaFree and cudaFreeHost wait for the device to go idle, which a running
// persistent kernel never does.
void release_when_idle(int device, std::vector<void*> device_ptrs, std::vector<void*> host_ptrs,
                       std::vector<cudaStream_t> streams);

struct LaneStats {
  uint32_t slots, inflight, inflight_max, running, tag, iter_in_prog, gen;
};

class Fabric {
 public:
  Fabric(int n_workers, size_t smem_bytes, int device, size_t pool_bytes = size_t(1) << 30);
  ~Fabric();
  Fabric(const Fabric&) = delete;
  Fabric& operator=(const Fabric&) = delete;

  void start();
  // Launch workers first_worker .. first_worker + workers - 1 on an external
  // stream (a green context's), with that context current around the
  // launch; several launches run at once, one per partition, and stop()
  // ends them all. The device state persists across stop() and a new
  // start, so running lanes resume where they were.
  void start_on(uint64_t stream, uint64_t context, int workers, int first_worker = 0);
  void stop();
  bool running() const { return running_; }
  // Longest back-off (ns) of a worker that found no tile: lower values
  // pick up a newly activated stage sooner and poll the rings more.
  void set_idle_cap(uint32_t ns);
  static constexpr uint32_t kDefaultIdleCapNs = 4096;
  // Stage-event tracing for profiling: the kernel records every stage's
  // activation and completion into ptr (device memory, capacity records of
  // two 64-bit words) until capacity is reached; capacity 0 turns it off.
  void set_trace(uint64_t ptr, uint32_t capacity);
  uint32_t trace_count();
  int device() const { return device_; }
  int num_workers() const { return n_workers_; }
  int num_sms() const { return n_sms_; }

  // tiles: n x 8 int64 words (see types.h TileDesc); stages: m x 8 int32
  // words (StageStatic); succ: successor list.
  int64_t upload(const int64_t* tiles, int64_t n_tiles, const int32_t* stages,
                 int64_t n_stages, const int32_t* succ, int64_t n_succ, int first_stage,
                 int reset_stage, int iterations);
  void release(int64_t handle);
  uint32_t generation(int64_t handle) const;

  // map[worker] = lane (255: none); caps[lane]; order: lanes to borrow from;
  // programs[lane] = handle (0: none), each handle on at most one lane (a
  // program's stage counters are its own). Returns the new epoch.
  uint64_t publish(const std::vector<int>& map, const std::vector<int>& caps,
                   const std::vector<int>& order, const std::vector<int64_t>& programs,
                   const std::vector<int>& forms = {});
  uint64_t epoch() const { return epoch_; }

  // Device writes that are safe while the kernel runs (staged copies on the
  // control stream), and pool-backed blocks for per-program operands.
  void write(uint64_t dst, const void* src, size_t bytes);
  uint64_t blob_alloc(size_t bytes);
  void blob_free(uint64_t addr);

  // Copy a tensor map into the fabric's device table; returns its device
  // address, which tiles name directly.
  uint64_t add_tensor_map(const CUtensorMap& map);

  // Inspection for tests and tools: device-to-host copies through the
  // control stream, safe while the kernel runs.
  void read(uint64_t src, void* dst, size_t bytes);
  struct ProgramInfo {
    uint64_t tiles, stages, runtime;
    uint32_t n_tiles, n_stages;
  };
  ProgramInfo program_info(int64_t handle) const;
  // Positions of a lane's stage ring (link: its DRAM-expert ring) and the
  // entries between head and tail, at most max_entries of them.
  struct RingView {
    uint64_t head, tail, reserve;
    std::vector<uint32_t> entries;
  };
  RingView lane_ring(int lane, bool link, size_t max_entries);
  // Each worker's record (see note_tile in runtime.cuh): the state word and
  // the device time the state began.
  struct WorkerTile {
    uint64_t state, ns;
  };
  std::vector<WorkerTile> worker_tiles();

  // In-kernel iteration timers: waits until the lane's iteration count
  // exceeds `start` by `count` (or the timeout) and returns samples of
  // (iterations, device ns at the end of that iteration). A sample covers
  // several iterations when they end faster than the host polls.
  std::vector<std::pair<uint64_t, uint64_t>> iteration_times(int lane, uint64_t start, uint64_t count,
                                                             double timeout_s) const;
  uint64_t iterations(int lane) const;

  const HostMirror& mirror() const { return *h_mirror_; }
  LaneStats lane_stats(int lane);
  void reset_inflight_max(int lane);
  DeviceState* device_state() const { return d_state_; }
  cudaStream_t control_stream() const { return ctrl_; }

 private:
  struct Program {
    ProgramDesc* desc;
    void* tiles;
    void* stages;
    void* runtime;
    void* succ;
    uint32_t gen;
    uint32_t n_tiles, n_stages;
  };
  void write_device(size_t offset, const void* src, size_t bytes);
  void copy_from_device(void* dst, const void* src, size_t bytes);
  // Device memory for programs comes from a pool allocated at construction:
  // while the persistent kernel runs, cudaMalloc and cudaFree could wait
  // for the device to go idle, which it never does.
  void* pool_alloc(size_t bytes);
  void pool_free(void* p);
  void copy_to_device(void* dst, const void* src, size_t bytes);

  int device_;
  int n_workers_;
  int n_sms_;
  size_t smem_;
  bool running_ = false;
  uint64_t epoch_ = 0;
  uint32_t next_gen_ = 1;
  DeviceState* d_state_ = nullptr;
  HostMirror* h_mirror_ = nullptr;
  HostMirror* d_mirror_ = nullptr;
  Snapshot* h_snap_ = nullptr;           // pinned staging
  unsigned long long* h_word_ = nullptr; // pinned staging for small writes
  cudaStream_t stream_ = nullptr;        // the persistent kernel
  cudaStream_t ctrl_ = nullptr;          // control copies
  // the running launches: stream, and green context (null for our own)
  std::vector<std::pair<cudaStream_t, CUcontext>> launches_;
  std::unordered_map<int64_t, Program> programs_;
  char* pool_ = nullptr;
  size_t pool_bytes_ = 0;
  std::map<size_t, size_t> free_;          // offset -> bytes
  std::unordered_map<void*, size_t> used_; // pointer -> bytes
  char* h_staging_ = nullptr;              // pinned bounce buffer for uploads
  static constexpr size_t kStagingBytes = size_t(16) << 20;
  CUtensorMap* d_maps_ = nullptr;         // device table of tensor maps
  size_t n_maps_ = 0;
  static constexpr size_t kMaxMaps = 1 << 16;
  std::mutex mu_;
};

}  // namespace monofab
