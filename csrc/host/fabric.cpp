#include "host/fabric.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

#include "fabric/launch.h"

namespace monofab {

namespace {
void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string("monofab: ") + what + ": " + cudaGetErrorString(e));
}

// Fabric kernels running per device, and the releases waiting for them. A
// fabric often outlives its last reference until Python's cycle collector
// runs, which may be while another fabric's kernel is running.
struct Deferred {
  int device;
  std::vector<void*> dev, host;
  std::vector<cudaStream_t> streams;
};
std::mutex g_mu;
std::unordered_map<int, int> g_running;
std::vector<Deferred> g_deferred;

void free_now(const Deferred& d) {
  int prev = -1;
  cudaGetDevice(&prev);
  cudaSetDevice(d.device);
  for (cudaStream_t s : d.streams) cudaStreamDestroy(s);
  for (void* p : d.dev) cudaFree(p);
  for (void* p : d.host) cudaFreeHost(p);
  if (prev >= 0) cudaSetDevice(prev);
}

// With g_mu held, so no fabric kernel starts on the device meanwhile.
void drain(int device) {
  auto keep = std::partition(g_deferred.begin(), g_deferred.end(),
                             [&](const Deferred& d) { return d.device != device; });
  for (auto it = keep; it != g_deferred.end(); ++it) free_now(*it);
  g_deferred.erase(keep, g_deferred.end());
}
}  // namespace

void release_when_idle(int device, std::vector<void*> device_ptrs, std::vector<void*> host_ptrs,
                       std::vector<cudaStream_t> streams) {
  std::lock_guard<std::mutex> g(g_mu);
  Deferred d{device, std::move(device_ptrs), std::move(host_ptrs), std::move(streams)};
  if (g_running[device] > 0)
    g_deferred.push_back(std::move(d));
  else
    free_now(d);
}

Fabric::Fabric(int n_workers, size_t smem_bytes, int device, size_t pool_bytes)
    : device_(device), smem_(smem_bytes), pool_bytes_(pool_bytes) {
  check(cudaSetDevice(device_), "cudaSetDevice");
  cudaDeviceProp prop;
  check(cudaGetDeviceProperties(&prop, device_), "cudaGetDeviceProperties");
  n_sms_ = prop.multiProcessorCount;
  n_workers_ = n_workers > 0 ? n_workers : n_sms_;
  if (n_workers_ > kMaxWorkers) throw std::runtime_error("monofab: too many workers");
  check(cudaMalloc(&d_state_, device_state_bytes()), "cudaMalloc state");
  check(cudaMemset(d_state_, 0, device_state_bytes()), "cudaMemset state");
  check(cudaHostAlloc(&h_mirror_, sizeof(HostMirror), cudaHostAllocMapped), "mirror");
  std::memset(h_mirror_, 0, sizeof(HostMirror));
  check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_mirror_), h_mirror_, 0), "mirror ptr");
  check(cudaHostAlloc(&h_snap_, sizeof(Snapshot), cudaHostAllocDefault), "snapshot staging");
  check(cudaHostAlloc(&h_word_, 64, cudaHostAllocDefault), "word staging");
  check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
  check(cudaStreamCreateWithFlags(&ctrl_, cudaStreamNonBlocking), "control stream");
  write_device(mirror_ptr_offset(), &d_mirror_, sizeof(d_mirror_));
  const unsigned int idle_cap = kDefaultIdleCapNs;
  write_device(idle_cap_offset(), &idle_cap, sizeof(idle_cap));
  check(cudaMalloc(&d_maps_, sizeof(CUtensorMap) * kMaxMaps), "tensor map table");
  check(cudaMalloc(&pool_, pool_bytes_), "program pool");
  free_[0] = pool_bytes_;
  check(cudaHostAlloc(&h_staging_, kStagingBytes, cudaHostAllocDefault), "upload staging");
}

void* Fabric::pool_alloc(size_t bytes) {
  bytes = (bytes + 255) & ~size_t(255);
  for (auto it = free_.begin(); it != free_.end(); ++it) {
    if (it->second >= bytes) {
      const size_t off = it->first, left = it->second - bytes;
      free_.erase(it);
      if (left) free_[off + bytes] = left;
      void* p = pool_ + off;
      used_[p] = bytes;
      return p;
    }
  }
  throw std::runtime_error("monofab: program pool exhausted");
}

void Fabric::pool_free(void* p) {
  auto it = used_.find(p);
  if (it == used_.end()) return;
  size_t off = static_cast<char*>(p) - pool_, bytes = it->second;
  used_.erase(it);
  auto nx = free_.lower_bound(off);
  if (nx != free_.end() && off + bytes == nx->first) {
    bytes += nx->second;
    nx = free_.erase(nx);
  }
  if (nx != free_.begin()) {
    auto pv = std::prev(nx);
    if (pv->first + pv->second == off) {
      pv->second += bytes;
      return;
    }
  }
  free_[off] = bytes;
}

void Fabric::copy_to_device(void* dst, const void* src, size_t bytes) {
  const char* s = static_cast<const char*>(src);
  char* d = static_cast<char*>(dst);
  while (bytes) {
    const size_t n = bytes < kStagingBytes ? bytes : kStagingBytes;
    std::memcpy(h_staging_, s, n);
    check(cudaMemcpyAsync(d, h_staging_, n, cudaMemcpyHostToDevice, ctrl_), "upload copy");
    check(cudaStreamSynchronize(ctrl_), "upload sync");
    s += n;
    d += n;
    bytes -= n;
  }
}

void Fabric::copy_from_device(void* dst, const void* src, size_t bytes) {
  const char* s = static_cast<const char*>(src);
  char* d = static_cast<char*>(dst);
  while (bytes) {
    const size_t n = bytes < kStagingBytes ? bytes : kStagingBytes;
    check(cudaMemcpyAsync(h_staging_, s, n, cudaMemcpyDeviceToHost, ctrl_), "read copy");
    check(cudaStreamSynchronize(ctrl_), "read sync");
    std::memcpy(d, h_staging_, n);
    s += n;
    d += n;
    bytes -= n;
  }
}

Fabric::~Fabric() {
  try {
    if (running_) stop();
  } catch (...) {
  }
  programs_.clear();
  release_when_idle(device_, {pool_, d_maps_, d_state_}, {h_staging_, h_mirror_, h_snap_, h_word_},
                    {stream_, ctrl_});
}

void Fabric::set_idle_cap(uint32_t ns) {
  std::lock_guard<std::mutex> g(mu_);
  write_device(idle_cap_offset(), &ns, sizeof(ns));
}

void Fabric::set_trace(uint64_t ptr, uint32_t capacity) {
  std::lock_guard<std::mutex> g(mu_);
  const unsigned int zero = 0;
  write_device(trace_cap_offset(), &zero, sizeof(zero));   // off while the buffer changes
  write_device(trace_ptr_offset(), &ptr, sizeof(ptr));
  write_device(trace_n_offset(), &zero, sizeof(zero));
  write_device(trace_cap_offset(), &capacity, sizeof(capacity));
}

uint32_t Fabric::trace_count() {
  std::lock_guard<std::mutex> g(mu_);
  unsigned int n = 0;
  copy_from_device(&n, reinterpret_cast<char*>(d_state_) + trace_n_offset(), sizeof(n));
  return n;
}

void Fabric::write_device(size_t offset, const void* src, size_t bytes) {
  // Stage through pinned memory so the copy is a true asynchronous DMA
  // that can run while the persistent kernel occupies every SM.
  if (bytes > 64) throw std::runtime_error("monofab: write_device too large");
  std::memcpy(h_word_, src, bytes);
  check(cudaMemcpyAsync(reinterpret_cast<char*>(d_state_) + offset, h_word_, bytes,
                        cudaMemcpyHostToDevice, ctrl_),
        "write_device");
  check(cudaStreamSynchronize(ctrl_), "write_device sync");
}

void Fabric::start() {
  std::lock_guard<std::mutex> g(mu_);
  if (running_) return;
  unsigned int zero = 0;
  write_device(shutdown_offset(), &zero, sizeof(zero));
  std::lock_guard<std::mutex> r(g_mu);
  check(launch_fabric(d_state_, n_workers_, smem_, stream_), "launch");
  g_running[device_]++;
  launches_.push_back({stream_, nullptr});
  running_ = true;
}

void Fabric::start_on(uint64_t stream, uint64_t context, int workers, int first_worker) {
  std::lock_guard<std::mutex> g(mu_);
  if (workers <= 0 || first_worker < 0 || first_worker + workers > n_workers_)
    throw std::runtime_error("monofab: bad worker range");
  if (!running_) {
    unsigned int zero = 0;
    write_device(shutdown_offset(), &zero, sizeof(zero));
  }
  CUcontext ctx = reinterpret_cast<CUcontext>(context);
  std::lock_guard<std::mutex> r(g_mu);
  if (ctx && cuCtxPushCurrent(ctx) != CUDA_SUCCESS) throw std::runtime_error("monofab: context push");
  const cudaError_t e = launch_fabric(d_state_, workers, smem_, reinterpret_cast<cudaStream_t>(stream),
                                      first_worker);
  if (ctx) {
    CUcontext prev;
    cuCtxPopCurrent(&prev);
  }
  check(e, "launch");
  g_running[device_]++;
  launches_.push_back({reinterpret_cast<cudaStream_t>(stream), ctx});
  running_ = true;
}

void Fabric::stop() {
  std::lock_guard<std::mutex> g(mu_);
  if (!running_) return;
  unsigned int one = 1;
  write_device(shutdown_offset(), &one, sizeof(one));
  cudaError_t e = cudaSuccess;
  for (auto& [stream, ctx] : launches_) {
    if (ctx) cuCtxPushCurrent(ctx);
    const cudaError_t le = cudaStreamSynchronize(stream);
    if (ctx) {
      CUcontext prev;
      cuCtxPopCurrent(&prev);
    }
    if (e == cudaSuccess) e = le;
  }
  running_ = false;
  {
    std::lock_guard<std::mutex> r(g_mu);
    g_running[device_] -= static_cast<int>(launches_.size());
    if (g_running[device_] == 0) drain(device_);
  }
  launches_.clear();
  check(e, "fabric exit");
}

int64_t Fabric::upload(const int64_t* tiles, int64_t n_tiles, const int32_t* stages,
                       int64_t n_stages, const int32_t* succ, int64_t n_succ, int first_stage,
                       int reset_stage, int iterations) {
  std::lock_guard<std::mutex> g(mu_);
  if (n_stages <= 0 || first_stage < 0 || first_stage >= n_stages || reset_stage < 0 ||
      reset_stage >= n_stages)
    throw std::runtime_error("monofab: bad program shape");
  Program p{};
  p.gen = next_gen_++;
  p.n_tiles = static_cast<uint32_t>(n_tiles);
  p.n_stages = static_cast<uint32_t>(n_stages);
  const size_t tb = sizeof(TileDesc) * (n_tiles > 0 ? n_tiles : 1);
  const size_t sb = sizeof(StageStatic) * n_stages;
  const size_t rb = sizeof(StageRuntime) * n_stages;
  const size_t cb = sizeof(uint32_t) * (n_succ > 0 ? n_succ : 1);
  p.tiles = pool_alloc(tb);
  p.stages = pool_alloc(sb);
  p.runtime = pool_alloc(rb);
  p.succ = pool_alloc(cb);
  p.desc = static_cast<ProgramDesc*>(pool_alloc(sizeof(ProgramDesc)));
  if (n_tiles > 0) copy_to_device(p.tiles, tiles, sizeof(TileDesc) * n_tiles);
  copy_to_device(p.stages, stages, sb);
  if (n_succ > 0) copy_to_device(p.succ, succ, sizeof(uint32_t) * n_succ);
  std::vector<StageRuntime> rt(n_stages);
  const StageStatic* st = reinterpret_cast<const StageStatic*>(stages);
  for (int64_t i = 0; i < n_stages; ++i) {
    rt[i].tag_next = 0;
    rt[i].done = 0;
    rt[i].deps = st[i].deps_init;
  }
  copy_to_device(p.runtime, rt.data(), rb);
  ProgramDesc d{};
  d.tiles = static_cast<TileDesc*>(p.tiles);
  d.stages = static_cast<StageStatic*>(p.stages);
  d.runtime = static_cast<StageRuntime*>(p.runtime);
  d.succ = static_cast<uint32_t*>(p.succ);
  d.n_tiles = static_cast<uint32_t>(n_tiles);
  d.n_stages = static_cast<uint32_t>(n_stages);
  d.first_stage = static_cast<uint32_t>(first_stage);
  d.reset_stage = static_cast<uint32_t>(reset_stage);
  d.iterations = static_cast<uint32_t>(iterations);
  d.gen = p.gen;
  copy_to_device(p.desc, &d, sizeof(d));
  const int64_t handle = reinterpret_cast<int64_t>(p.desc);
  programs_[handle] = p;
  return handle;
}

void Fabric::release(int64_t handle) {
  std::lock_guard<std::mutex> g(mu_);
  auto it = programs_.find(handle);
  if (it == programs_.end()) return;
  pool_free(it->second.desc);
  pool_free(it->second.tiles);
  pool_free(it->second.stages);
  pool_free(it->second.runtime);
  pool_free(it->second.succ);
  programs_.erase(it);
}

uint32_t Fabric::generation(int64_t handle) const {
  auto it = programs_.find(handle);
  return it == programs_.end() ? 0 : it->second.gen;
}

uint64_t Fabric::publish(const std::vector<int>& map, const std::vector<int>& caps,
                         const std::vector<int>& order, const std::vector<int64_t>& programs,
                         const std::vector<int>& forms) {
  std::lock_guard<std::mutex> g(mu_);
  Snapshot& s = *h_snap_;
  std::memset(&s, 0, sizeof(s));
  s.epoch = epoch_ + 1;
  s.n_workers = n_workers_;
  s.n_lanes = static_cast<uint32_t>(programs.size());
  if (s.n_lanes > kMaxLanes) throw std::runtime_error("monofab: too many lanes");
  for (int w = 0; w < kMaxWorkers; ++w)
    s.map[w] = w < static_cast<int>(map.size()) ? static_cast<uint8_t>(map[w]) : kNoLane;
  for (int l = 0; l < kMaxLanes; ++l) {
    s.cap[l] = l < static_cast<int>(caps.size()) ? caps[l] : 0;
    s.order[l] = l < static_cast<int>(order.size()) ? order[l] : kNoLane;
    s.form[l] = l < static_cast<int>(forms.size()) ? forms[l] : 64;
    if (l < static_cast<int>(programs.size()) && programs[l] != 0) {
      auto it = programs_.find(programs[l]);
      if (it == programs_.end()) throw std::runtime_error("monofab: unknown program");
      // a program's stage counters are its own, so two lanes would race on them
      for (int k = 0; k < l; ++k)
        if (programs[k] == programs[l])
          throw std::runtime_error("monofab: a program runs on one lane at a time");
      s.prog[l] = it->second.desc;
      s.prog_gen[l] = it->second.gen;
    }
  }
  const uint64_t e = epoch_ + 1;
  check(cudaMemcpyAsync(reinterpret_cast<char*>(d_state_) + snapshot_offset(e & 1), h_snap_,
                        sizeof(Snapshot), cudaMemcpyHostToDevice, ctrl_),
        "snapshot copy");
  *h_word_ = e;
  check(cudaMemcpyAsync(reinterpret_cast<char*>(d_state_) + published_offset(), h_word_,
                        sizeof(unsigned long long), cudaMemcpyHostToDevice, ctrl_),
        "epoch copy");
  check(cudaStreamSynchronize(ctrl_), "publish sync");
  epoch_ = e;
  return e;
}

LaneStats Fabric::lane_stats(int lane) {
  std::lock_guard<std::mutex> g(mu_);
  const LaneCounters c = lane_counter_offsets();
  const size_t base = lane_offset(lane);
  unsigned int v[7];
  const size_t offs[7] = {c.slots, c.inflight, c.inflight_max, c.running,
                          c.tag, c.iter_in_prog, c.gen};
  unsigned int* staging = reinterpret_cast<unsigned int*>(h_word_);
  for (int i = 0; i < 7; ++i) {
    check(cudaMemcpyAsync(staging, reinterpret_cast<char*>(d_state_) + base + offs[i],
                          sizeof(unsigned int), cudaMemcpyDeviceToHost, ctrl_),
          "lane stat");
    check(cudaStreamSynchronize(ctrl_), "lane stat sync");
    v[i] = *staging;
  }
  return LaneStats{v[0], v[1], v[2], v[3], v[4], v[5], v[6]};
}

uint64_t Fabric::iterations(int lane) const {
  if (lane < 0 || lane >= kMaxLanes) throw std::runtime_error("monofab: bad lane");
  return reinterpret_cast<const volatile LaneMirror*>(&h_mirror_->lane[lane])->iterations;
}

std::vector<std::pair<uint64_t, uint64_t>> Fabric::iteration_times(int lane, uint64_t start,
                                                                   uint64_t count,
                                                                   double timeout_s) const {
  if (lane < 0 || lane >= kMaxLanes) throw std::runtime_error("monofab: bad lane");
  const volatile LaneMirror* m = &h_mirror_->lane[lane];
  std::vector<std::pair<uint64_t, uint64_t>> out;
  uint64_t seen = start;
  const auto t0 = std::chrono::steady_clock::now();
  while (seen < start + count) {
    const uint64_t it = m->iterations;
    if (it > seen) {
      // iter_ns is written before the count; re-read the count to pair them
      std::atomic_thread_fence(std::memory_order_acquire);
      const uint64_t ns = m->iter_ns;
      std::atomic_thread_fence(std::memory_order_acquire);
      if (m->iterations != it) continue;
      out.emplace_back(it, ns);
      seen = it;
      continue;
    }
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) break;
  }
  return out;
}

void Fabric::read(uint64_t src, void* dst, size_t bytes) {
  std::lock_guard<std::mutex> g(mu_);
  copy_from_device(dst, reinterpret_cast<const void*>(src), bytes);
}

Fabric::ProgramInfo Fabric::program_info(int64_t handle) const {
  auto it = programs_.find(handle);
  if (it == programs_.end()) throw std::runtime_error("monofab: unknown program");
  const Program& p = it->second;
  return ProgramInfo{reinterpret_cast<uint64_t>(p.tiles), reinterpret_cast<uint64_t>(p.stages),
                     reinterpret_cast<uint64_t>(p.runtime), p.n_tiles, p.n_stages};
}

Fabric::RingView Fabric::lane_ring(int lane, bool link, size_t max_entries) {
  if (lane < 0 || lane >= kMaxLanes) throw std::runtime_error("monofab: bad lane");
  std::lock_guard<std::mutex> g(mu_);
  const LaneCounters c = lane_counter_offsets();
  const char* base = reinterpret_cast<const char*>(d_state_) + lane_offset(lane);
  RingView v{0, 0, 0, {}};
  copy_from_device(&v.head, base + (link ? c.link_head : c.ring_head), sizeof(v.head));
  copy_from_device(&v.tail, base + (link ? c.link_tail : c.ring_tail), sizeof(v.tail));
  copy_from_device(&v.reserve, base + (link ? c.link_reserve : c.ring_reserve), sizeof(v.reserve));
  if (v.tail > v.head) {
    std::vector<uint32_t> ring(kRingCap);
    copy_from_device(ring.data(), base + (link ? c.link : c.ring), sizeof(uint32_t) * kRingCap);
    for (uint64_t p = v.head; p < v.tail && v.entries.size() < max_entries; ++p)
      v.entries.push_back(ring[p & (kRingCap - 1)]);
  }
  return v;
}

std::vector<Fabric::WorkerTile> Fabric::worker_tiles() {
  std::lock_guard<std::mutex> g(mu_);
  std::vector<unsigned long long> state(n_workers_), ns(n_workers_);
  const char* base = reinterpret_cast<const char*>(d_state_);
  copy_from_device(state.data(), base + worker_tile_offset(), sizeof(unsigned long long) * n_workers_);
  copy_from_device(ns.data(), base + worker_tile_ns_offset(), sizeof(unsigned long long) * n_workers_);
  std::vector<WorkerTile> out(n_workers_);
  for (int w = 0; w < n_workers_; ++w) out[w] = WorkerTile{state[w], ns[w]};
  return out;
}

void Fabric::reset_inflight_max(int lane) {
  const unsigned int zero = 0;
  write_device(lane_offset(lane) + lane_counter_offsets().inflight_max, &zero, sizeof(zero));
}

uint64_t Fabric::add_tensor_map(const CUtensorMap& map) {
  std::lock_guard<std::mutex> g(mu_);
  if (n_maps_ >= kMaxMaps) throw std::runtime_error("monofab: tensor map table full");
  CUtensorMap* slot = d_maps_ + n_maps_++;
  std::memcpy(h_staging_, &map, sizeof(CUtensorMap));
  check(cudaMemcpyAsync(slot, h_staging_, sizeof(CUtensorMap), cudaMemcpyHostToDevice, ctrl_), "map copy");
  check(cudaStreamSynchronize(ctrl_), "map copy sync");
  return reinterpret_cast<uint64_t>(slot);
}

void Fabric::write(uint64_t dst, const void* src, size_t bytes) {
  std::lock_guard<std::mutex> g(mu_);
  copy_to_device(reinterpret_cast<void*>(dst), src, bytes);
}

uint64_t Fabric::blob_alloc(size_t bytes) {
  std::lock_guard<std::mutex> g(mu_);
  return reinterpret_cast<uint64_t>(pool_alloc(bytes));
}

void Fabric::blob_free(uint64_t addr) {
  std::lock_guard<std::mutex> g(mu_);
  pool_free(reinterpret_cast<void*>(addr));
}

}  // namespace monofab
