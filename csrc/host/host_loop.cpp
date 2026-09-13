#include "host/host_loop.h"

#include <chrono>
#include <stdexcept>
#include <string>

#include "host/fabric.h"

namespace monofab {

namespace {
void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("host loop: ") + what + ": " + cudaGetErrorString(e));
}
}  // namespace

HostLoop::HostLoop(Fabric* fab, int layers, int experts, size_t w13_bytes, size_t w2_bytes,
                   std::vector<uint64_t> host_w13, std::vector<uint64_t> host_w2, int host_region)
    : fab_(fab), layers_(layers), experts_(experts), host_region_(host_region),
      w13_bytes_(w13_bytes), w2_bytes_(w2_bytes), host_w13_(std::move(host_w13)),
      host_w2_(std::move(host_w2)) {
  device_ = fab->device();
  check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
  check(cudaHostAlloc(&h_entry_, 4096, cudaHostAllocDefault), "entry staging");
}

HostLoop::~HostLoop() {
  stop();
  for (auto& s : lanes_) cudaEventDestroy(s.done);
  release_when_idle(device_, {}, {h_entry_}, {stream_});
}

void HostLoop::add_lane(const StagingLaneConfig& cfg) {
  if (run_) throw std::runtime_error("host loop: add lanes before start");
  LaneState s;
  s.cfg = cfg;
  check(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming), "event");
  lanes_.push_back(std::move(s));
}

void HostLoop::start(int poll_us) {
  if (run_.exchange(true)) return;
  thread_ = std::thread([this, poll_us] { loop(poll_us); });
}

void HostLoop::stop() {
  if (!run_.exchange(false)) return;
  if (thread_.joinable()) thread_.join();
  cudaStreamSynchronize(stream_);
}

HostLoopStats HostLoop::stats() const {
  return HostLoopStats{copies_.load(), bytes_.load(), flips_.load(), restores_.load(), polls_.load()};
}

// Table entries are written by the copy engine, ordered on the same stream
// after the expert copy they publish; each entry is one aligned 8-byte write.
void HostLoop::write_entry(LaneState& s, int layer, int expert, int region, int slot) {
  int* e = h_entry_ + 2 * (entry_ring_++ % 256);
  e[0] = region;
  e[1] = slot;
  const uint64_t dst = s.cfg.tables + (static_cast<uint64_t>(layer) * experts_ + expert) * 8;
  check(cudaMemcpyAsync(reinterpret_cast<void*>(dst), e, 8, cudaMemcpyHostToDevice, stream_),
        "table write");
}

void HostLoop::restore_half(LaneState& s, int half) {
  const int layer = s.staged_layer[half];
  if (layer >= 0)
    for (auto& fe : s.flipped[half]) {
      write_entry(s, layer, fe.first, host_region_, fe.first);
      restores_++;
    }
  s.flipped[half].clear();
  s.staged_layer[half] = -1;
  s.filled[half] = 0;
}

void HostLoop::step(LaneState& s) {
  const LaneMirror& m = fab_->mirror().lane[s.cfg.lane];
  // acquire ordering between the reads: the device publishes running
  // before gen, and a host such as Grace may reorder plain loads
  const unsigned long long mark = *reinterpret_cast<const volatile unsigned long long*>(&m.mark);
  std::atomic_thread_fence(std::memory_order_acquire);
  const unsigned long long gen = *reinterpret_cast<const volatile unsigned long long*>(&m.gen);
  std::atomic_thread_fence(std::memory_order_acquire);
  const unsigned long long running = *reinterpret_cast<const volatile unsigned long long*>(&m.running);

  // a finished copy: publish its table entry
  if (s.pending && cudaEventQuery(s.done) == cudaSuccess) {
    s.pending = false;
    const int layer = s.staged_layer[s.pend_half];
    if (layer >= 0) {
      write_entry(s, layer, s.pend_expert, s.cfg.region, s.pend_slot);
      s.flipped[s.pend_half].push_back({s.pend_expert, s.pend_slot});
      flips_++;
    }
  }
  if (gen != s.gen || !running) {   // new program or lane idle: start over
    if (!s.pending) {
      restore_half(s, 0);
      restore_half(s, 1);
      s.gen = gen;
    }
    if (!running) return;
  }
  const int layer = static_cast<int>(mark >> 8);
  const int phase = static_cast<int>(mark & 0xff);   // 1 attention, 2 MoE window, 3 layer done
  const int cur = phase == 3 ? layer + 1 : layer;     // the layer the lane works on now
  // halves of layers that are done are free again
  for (int h = 0; h < 2; ++h)
    if (s.staged_layer[h] >= 0 && s.staged_layer[h] < cur && !(s.pending && s.pend_half == h))
      restore_half(s, h);
  if (s.pending) return;
  // stage the current layer if its MoE window has not begun, else the next
  int target = cur;
  if (phase == 2 && layer == cur) target = cur + 1;
  if (target >= layers_) return;
  const int half = target & 1;
  if (s.staged_layer[half] != target) {
    if (s.staged_layer[half] >= 0) return;   // still in use by an earlier layer
    s.staged_layer[half] = target;
    s.filled[half] = 0;
  }
  const auto& order = s.cfg.order[target];
  if (s.filled[half] >= static_cast<int>(order.size()) || s.filled[half] >= s.cfg.slots) return;
  const int e = order[s.filled[half]];
  const int slot = half * s.cfg.slots + s.filled[half];
  s.filled[half]++;
  const uint64_t src13 = host_w13_[target] + static_cast<uint64_t>(e) * w13_bytes_;
  const uint64_t src2 = host_w2_[target] + static_cast<uint64_t>(e) * w2_bytes_;
  check(cudaMemcpyAsync(reinterpret_cast<void*>(s.cfg.w13_dst + slot * w13_bytes_),
                        reinterpret_cast<const void*>(src13), w13_bytes_, cudaMemcpyHostToDevice,
                        stream_), "stage w13");
  check(cudaMemcpyAsync(reinterpret_cast<void*>(s.cfg.w2_dst + slot * w2_bytes_),
                        reinterpret_cast<const void*>(src2), w2_bytes_, cudaMemcpyHostToDevice,
                        stream_), "stage w2");
  check(cudaEventRecord(s.done, stream_), "event record");
  s.pending = true;
  s.pend_half = half;
  s.pend_expert = e;
  s.pend_slot = slot;
  copies_++;
  bytes_ += w13_bytes_ + w2_bytes_;
}

void HostLoop::loop(int poll_us) {
  while (run_) {
    for (auto& s : lanes_) step(s);
    polls_++;
    std::this_thread::sleep_for(std::chrono::microseconds(poll_us));
  }
  for (auto& s : lanes_) {
    if (s.pending) cudaEventSynchronize(s.done);
    s.pending = false;
    restore_half(s, 0);
    restore_half(s, 1);
  }
}

}  // namespace monofab
