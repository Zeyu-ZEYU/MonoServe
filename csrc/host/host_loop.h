// The host loop: a C++ thread that follows the fabric's progress and drives
// the copy engine.
//
// It polls the lanes' marks (layer and phase) in pinned host memory. While a
// prefill lane is in a layer's attention window, when the lane's own link
// share is idle, it copies that layer's experts (the next ones by the
// prefill profile that the hot tier does not hold) from pinned CPU DRAM into
// the half of the lane's staging buffer reserved for the layer, one
// expert-sized copy at a time on a dedicated stream, and flips each expert's
// entry in the lane's indirection table once its copy has landed. When the
// layer's MoE window completes, the entries go back to CPU DRAM and the half
// is refilled for the layer after next. A late copy is harmless: the
// expander finds the old entry and the tile reads over the link instead.
#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

namespace monofab {

class Fabric;

struct StagingLaneConfig {
  int lane;
  int region;                 // region id of this lane's staging buffer
  uint64_t w13_dst, w2_dst;   // staging slots: [slots][w13_bytes] and [slots][w2_bytes]
  int slots;                  // slots per half (one layer's worth)
  uint64_t tables;            // this lane's tables: int32 [layers][experts][2]
  std::vector<std::vector<int>> order;   // per layer: experts to stage, best first
};

struct HostLoopStats {
  uint64_t copies, bytes, flips, restores, polls;
};

class HostLoop {
 public:
  HostLoop(Fabric* fab, int layers, int experts, size_t w13_bytes, size_t w2_bytes,
           std::vector<uint64_t> host_w13, std::vector<uint64_t> host_w2, int host_region);
  ~HostLoop();
  void add_lane(const StagingLaneConfig& cfg);
  void start(int poll_us);
  void stop();
  HostLoopStats stats() const;

 private:
  struct LaneState {
    StagingLaneConfig cfg;
    unsigned long long gen = 0;
    int staged_layer[2] = {-1, -1};      // layer held by each half
    int filled[2] = {0, 0};              // experts copied into each half
    std::vector<std::pair<int, int>> flipped[2];   // (expert, slot) per half
    bool pending = false;                // a copy is in flight
    int pend_half = 0, pend_expert = -1, pend_slot = -1;
    cudaEvent_t done{};
  };
  void loop(int poll_us);
  void step(LaneState& s);
  void restore_half(LaneState& s, int half);
  void write_entry(LaneState& s, int layer, int expert, int region, int slot);

  Fabric* fab_;
  int layers_, experts_, host_region_;
  size_t w13_bytes_, w2_bytes_;
  std::vector<uint64_t> host_w13_, host_w2_;
  std::vector<LaneState> lanes_;
  cudaStream_t stream_{};
  int* h_entry_ = nullptr;             // pinned staging for table writes
  int entry_ring_ = 0;
  int device_ = 0;
  std::thread thread_;
  std::atomic<bool> run_{false};
  std::atomic<uint64_t> copies_{0}, bytes_{0}, flips_{0}, restores_{0}, polls_{0};
};

}  // namespace monofab
