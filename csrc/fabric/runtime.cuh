// Device side of the MonoFab runtime: lane state, stage rings, link slots,
// epochs, and the scheduling loop of a persistent worker.
//
// Every SM runs one worker. Thread 0 of a worker schedules: it reads the
// epoch (plan version) and its SM-to-lane assignment, then takes a tile
// from its own lane (a DRAM-expert tile if it holds or can acquire one of
// the lane's link slots, otherwise an ungated tile), and from another lane
// when its own has nothing ready. All threads then run the tile, and
// thread 0 records the completion, which may activate successor stages.
#pragma once
#include "fabric/sync.cuh"
#include "fabric/types.h"

namespace monofab {

constexpr uint32_t kStageBits = 20;
constexpr uint32_t kStageMask = (1u << kStageBits) - 1;
constexpr uint32_t kTagMask = 0xfffu;
constexpr int kScanDepth = 16;
constexpr int kWorklistCap = 512;
// Tiles taken per claim: one per kBatchDivisor tiles left in the stage, at
// most kMaxBatch, so large stages are handed out in fewer compare-and-swaps.
constexpr uint32_t kMaxBatch = 8;
constexpr uint32_t kBatchDivisor = 128;

struct alignas(128) PadU64 { unsigned long long v; unsigned long long pad[15]; };
struct alignas(128) PadU32 { unsigned int v; unsigned int pad[31]; };

// A ring of activated stage ids. Producers reserve a position, write the
// entry, and publish positions in reservation order; consumers scan from
// head and advance it past stale or exhausted entries.
struct LaneState {
  PadU64 ring_head, ring_reserve, ring_tail;   // ungated stages
  PadU64 link_head, link_reserve, link_tail;   // DRAM-expert stages
  PadU32 slots;          // link slots currently held
  PadU32 inflight;       // DRAM-expert tiles executing now
  PadU32 inflight_max;   // high-water mark of inflight
  PadU32 running;        // 1 while the lane has an active program
  PadU32 tag;            // iteration tag of the current activation wave
  PadU32 iter_in_prog;   // iterations completed by the active program
  PadU64 prog;           // const ProgramDesc* of the active program
  PadU32 gen;            // generation of the active program
  unsigned int ring[kRingCap];
  unsigned int link[kRingCap];
};

struct DeviceState {
  LaneState lane[kMaxLanes];
  Snapshot snap[2];
  PadU64 published;      // current epoch; written last by the host
  PadU32 shutdown;
  PadU32 idle_cap;       // longest back-off (ns) of a worker that found no tile (host-set)
  PadU32 trace_cap;      // stage-event records the trace buffer holds (0: tracing off)
  PadU32 trace_n;        // records written so far
  PadU64 trace_ptr;      // unsigned long long [trace_cap][2] (Fabric::set_trace)
  HostMirror* mirror;    // pinned host memory, or null
  // what each worker is doing (note_tile), for hang diagnosis
  unsigned long long worker_tile[kMaxWorkers];
  unsigned long long worker_tile_ns[kMaxWorkers];
};

// Stage events for profiling: activation (0, the last predecessor
// completed) and completion (1, the last tile completed), each with the
// iteration tag and the device time.
__device__ __forceinline__ void trace_stage(DeviceState& S, uint32_t lane, uint32_t s,
                                            uint32_t event, uint32_t tag) {
  const uint32_t cap = ld_volatile(&S.trace_cap.v);
  if (cap == 0) return;
  const uint32_t i = atomicAdd(&S.trace_n.v, 1u);
  if (i >= cap) return;
  unsigned long long* r = reinterpret_cast<unsigned long long*>(ld_volatile(&S.trace_ptr.v)) + 2ull * i;
  r[0] = static_cast<unsigned long long>(s) | (static_cast<unsigned long long>(lane) << 32) |
         (static_cast<unsigned long long>(event) << 40) |
         (static_cast<unsigned long long>(tag & kTagMask) << 48);
  r[1] = globaltimer_ns();
}

enum : uint32_t { kCmdRun = 0, kCmdIdle = 1, kCmdExit = 2 };

// This worker's index: its block plus the first worker of its launch (the
// /KF ablation launches the kernel once per green-context partition).
__device__ __forceinline__ uint32_t& worker_slot() {
  __shared__ uint32_t w;
  return w;
}
__device__ __forceinline__ uint32_t worker_index() { return worker_slot(); }

// A worker's record in DeviceState::worker_tile (Fabric::worker_tiles):
// running a tile or completing a claim (the flags below, or neither), and
// the tile's lane, kind, stage, and index; worker_tile_ns has the device
// time the state began. Plain stores by thread 0, read by the host.
constexpr unsigned long long kWorkerRunning = 1ull << 63;
constexpr unsigned long long kWorkerCompleting = 1ull << 62;

// Scheduling state of one worker, in shared memory.
struct WorkerShared {
  TileDesc tile;
  const ProgramDesc* prog;
  unsigned long long epoch;
  uint32_t stage;
  uint32_t tile_lane;
  uint32_t cmd;
  uint32_t link_tile;
  uint32_t batch;        // tiles of the current claim
  uint32_t tile_left;    // of them, still to run after the current one
  uint32_t tile_index;   // the current tile
  uint32_t more;         // the next tile of the claim runs without scheduling
  uint32_t lane;
  uint32_t n_lanes;
  uint32_t slot_lane;
  uint32_t idle_ns;
  uint32_t cap[kMaxLanes];
  uint32_t order[kMaxLanes];
  uint32_t prog_gen[kMaxLanes];
  ProgramDesc* prog_snap[kMaxLanes];
};

struct Worklist {
  uint32_t v[kWorklistCap];
  int n;
  __device__ void push(uint32_t s) { if (n < kWorklistCap) v[n++] = s; else __trap(); }
  __device__ uint32_t pop() { return v[--n]; }
};

__device__ __forceinline__ uint32_t make_entry(uint32_t stage, uint32_t tag) {
  return ((tag & kTagMask) << kStageBits) | (stage & kStageMask);
}

// ---------------------------------------------------------------------------
// Host mirror
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mirror_store(unsigned long long* p, unsigned long long v) {
  st_release_sys(p, v);
}

__device__ void mirror_lane(DeviceState& S, uint32_t lane, int what, unsigned long long v) {
  if (!S.mirror) return;
  LaneMirror& m = S.mirror->lane[lane];
  switch (what) {
    case 0: mirror_store(&m.iterations, v); break;
    case 1: mirror_store(&m.gen, v); break;
    case 2: mirror_store(&m.running, v); break;
    case 3:
      mirror_store(&m.mark_ns, globaltimer_ns());
      mirror_store(&m.mark, v);
      break;
  }
}

// ---------------------------------------------------------------------------
// Stage rings
// ---------------------------------------------------------------------------
// Append m entries with one reservation, and publish them once every
// earlier reservation is published.
__device__ void ring_push_n(LaneState& L, bool link, const uint32_t* entries, int m) {
  if (m == 0) return;
  PadU64& reserve = link ? L.link_reserve : L.ring_reserve;
  PadU64& tail = link ? L.link_tail : L.ring_tail;
  unsigned int* ids = link ? L.link : L.ring;
  const unsigned long long pos = atomicAdd(&reserve.v, static_cast<unsigned long long>(m));
  for (int i = 0; i < m; ++i) ids[(pos + i) & (kRingCap - 1)] = entries[i];
  unsigned int ns = 32;
  while (ld_acquire(&tail.v) != pos) {
    __nanosleep(ns);
    if (ns < 512) ns <<= 1;
  }
  st_release(&tail.v, pos + m);
}

__device__ void ring_push(LaneState& L, bool link, uint32_t entry) { ring_push_n(L, link, &entry, 1); }

// Entries a drain collects before publishing them together.
struct PushBatch {
  static constexpr int kCap = 64;
  uint32_t e[kCap];
  int n;
};

// Try to take the next tiles of the stage an entry refers to: n tiles from
// index. The claim is valid only if the stage's runtime still carries the
// entry's tag.
__device__ bool claim(const ProgramDesc* P, uint32_t entry, uint32_t& stage, uint32_t& index,
                      uint32_t& n) {
  const uint32_t s = entry & kStageMask;
  const uint32_t tag = entry >> kStageBits;
  if (s >= P->n_stages) return false;
  uint32_t count = ld_volatile(&P->stages[s].count);
  unsigned long long* tn = &P->runtime[s].tag_next;
  // check without writing: a stale entry leaves the stage alone
  const unsigned long long v = ld_acquire(tn);
  const uint32_t next = static_cast<uint32_t>(v);
  if ((static_cast<uint32_t>(v >> 32) & kTagMask) != tag || next >= count) return false;
  const uint32_t left = count - next;
  const uint32_t k = min(left, min(kMaxBatch, max(1u, left / kBatchDivisor)));
  // one fetch-and-add, however many workers contend: overshooting the count
  // is harmless (the next activation resets the word), and a newer
  // activation of the stage (another tag, same program) hands out its own
  // tiles, which are then this claim's
  const unsigned long long old = atomicAdd(tn, static_cast<unsigned long long>(k));
  const uint32_t ot = static_cast<uint32_t>(old >> 32) & kTagMask;
  const uint32_t on = static_cast<uint32_t>(old);
  if (ot == 0) return false;   // never activated since upload
  if (ot != tag) count = ld_volatile(&P->stages[s].count);
  if (on >= count) return false;
  stage = s;
  index = ld_volatile(&P->stages[s].first) + on;
  n = min(k, count - on);
  return true;
}

__device__ bool take_from(LaneState& L, bool link, const ProgramDesc* P, uint32_t& stage,
                          uint32_t& index, uint32_t& n) {
  PadU64& head = link ? L.link_head : L.ring_head;
  PadU64& tail = link ? L.link_tail : L.ring_tail;
  const unsigned int* ids = link ? L.link : L.ring;
  unsigned long long h = ld_acquire(&head.v);
  const unsigned long long t = ld_acquire(&tail.v);
  for (unsigned long long p = h; p < t && p < h + kScanDepth; ++p) {
    const uint32_t e = ld_volatile(&ids[p & (kRingCap - 1)]);
    if (claim(P, e, stage, index, n)) return true;
    if (p == h) {   // the head entry is stale or exhausted: step past it
      if (atomicCAS(&head.v, h, h + 1) == h) h = h + 1;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// Activation, completion, iterations
// ---------------------------------------------------------------------------
// Arm stage s for this activation. A stage without tiles goes to the
// worklist and completes at once (false); otherwise its tiles are open under
// the tag, and entry and link say what to publish on which ring.
__device__ bool arm(DeviceState& S, uint32_t lane, const ProgramDesc* P, uint32_t s, uint32_t tag,
                    Worklist& wl, uint32_t& entry, bool& link) {
  trace_stage(S, lane, s, 0, tag);
  const uint32_t count = ld_volatile(&P->stages[s].count);
  if (count == 0) {
    wl.push(s);
    return false;
  }
  atomicExch(&P->runtime[s].tag_next, static_cast<unsigned long long>(tag & kTagMask) << 32);
  link = (ld_volatile(&P->stages[s].flags) & kStageLink) != 0;
  entry = make_entry(s, tag);
  return true;
}

__device__ void activate(DeviceState& S, uint32_t lane, const ProgramDesc* P, uint32_t s,
                         uint32_t tag, Worklist& wl) {
  uint32_t entry;
  bool link;
  if (arm(S, lane, P, s, tag, wl, entry, link)) ring_push(S.lane[lane], link, entry);
}

__device__ void latest_program(DeviceState& S, uint32_t lane, ProgramDesc*& p, uint32_t& gen) {
  unsigned long long e = ld_acquire(&S.published.v);
  for (;;) {
    const Snapshot& sn = S.snap[e & 1];
    p = ld_volatile(&sn.prog[lane]);
    gen = ld_volatile(&sn.prog_gen[lane]);
    const unsigned long long e2 = ld_acquire(&S.published.v);
    if (e2 == e) return;
    e = e2;
  }
}

// Start an iteration of P on the lane: a new tag, then the first stage.
__device__ void begin_iteration(DeviceState& S, uint32_t lane, const ProgramDesc* P,
                                uint32_t gen, bool fresh, Worklist& wl) {
  LaneState& L = S.lane[lane];
  if (fresh) {
    // All entries of the previous program are exhausted: drop them.
    atomicExch(&L.ring_head.v, ld_acquire(&L.ring_tail.v));
    atomicExch(&L.link_head.v, ld_acquire(&L.link_tail.v));
    L.iter_in_prog.v = 0;
    st_release(&L.prog.v, reinterpret_cast<unsigned long long>(P));
    st_release(&L.gen.v, gen);
    mirror_lane(S, lane, 1, gen);
  }
  uint32_t t = ld_volatile(&L.tag.v) + 1;
  if ((t & kTagMask) == 0) ++t;   // tag 0 marks never-activated stages
  st_release(&L.tag.v, t);
  activate(S, lane, P, P->first_stage, t & kTagMask, wl);
}

__device__ const ProgramDesc* end_iteration(DeviceState& S, uint32_t lane,
                                            const ProgramDesc* P, Worklist& wl) {
  LaneState& L = S.lane[lane];
  // The reset stage's tiles re-armed every other stage; re-arm this one.
  StageRuntime& rr = P->runtime[P->reset_stage];
  rr.done = 0;
  rr.deps = ld_volatile(&P->stages[P->reset_stage].deps_init);
  const uint32_t it = L.iter_in_prog.v + 1;
  L.iter_in_prog.v = it;
  if (S.mirror) {
    // in-kernel timer: the end time is written before the count
    mirror_store(&S.mirror->lane[lane].iter_ns, globaltimer_ns());
    mirror_lane(S, lane, 0, ld_volatile(&S.mirror->lane[lane].iterations) + 1);
  }

  ProgramDesc* np;
  uint32_t ng;
  latest_program(S, lane, np, ng);
  if (np != nullptr && ng != ld_volatile(&L.gen.v)) {
    begin_iteration(S, lane, np, ng, true, wl);
    return np;
  }
  if (P->iterations != 0 && it >= P->iterations) {
    st_release(&L.running.v, 0u);
    mirror_lane(S, lane, 2, 0);
    return P;
  }
  begin_iteration(S, lane, P, ld_volatile(&L.gen.v), false, wl);
  return P;
}

// Process completed stages: notify successors, run iteration boundaries.
__device__ void drain(DeviceState& S, uint32_t lane, const ProgramDesc* P, Worklist& wl) {
  LaneState& L = S.lane[lane];
  while (wl.n > 0) {
    const uint32_t s = wl.pop();
    trace_stage(S, lane, s, 1, ld_volatile(&L.tag.v));
    const StageStatic& st = P->stages[s];
    const uint32_t flags = ld_volatile(&st.flags);
    if (flags & kStageMark) mirror_lane(S, lane, 3, ld_volatile(&st.mark));
    if (flags & kStageReset) {
      P = end_iteration(S, lane, P, wl);
      continue;
    }
    const uint32_t tag = ld_volatile(&L.tag.v) & kTagMask;
    const uint32_t sf = ld_volatile(&st.succ_first);
    const uint32_t sc = ld_volatile(&st.succ_count);
    // the successors that become ready are published together, one
    // reservation per ring (an expander's fan-out opens many at once)
    PushBatch ring_b, link_b;
    ring_b.n = link_b.n = 0;
    for (uint32_t k = 0; k < sc; ++k) {
      const uint32_t j = P->succ[sf + k];
      // a skipped stage's chain is out of this iteration; the expander took
      // its share off the join stage's count
      if (ld_volatile(&P->stages[j].flags) & kStageSkip) continue;
      if (atomicSub(&P->runtime[j].deps, 1u) != 1u) continue;
      uint32_t entry;
      bool link;
      if (!arm(S, lane, P, j, tag, wl, entry, link)) continue;
      PushBatch& b = link ? link_b : ring_b;
      b.e[b.n++] = entry;
      if (b.n == PushBatch::kCap) {
        ring_push_n(L, link, b.e, b.n);
        b.n = 0;
      }
    }
    ring_push_n(L, false, ring_b.e, ring_b.n);
    ring_push_n(L, true, link_b.e, link_b.n);
  }
}

__device__ void complete_tile(DeviceState& S, uint32_t lane, const ProgramDesc* P, uint32_t s,
                              uint32_t n = 1) {
  const unsigned int d = atomicAdd(&P->runtime[s].done, n) + n;
  if (d == ld_volatile(&P->stages[s].count)) {
    // acquire the other workers' tiles of the stage before releasing its
    // successors
    __threadfence();
    Worklist wl;
    wl.n = 0;
    wl.push(s);
    drain(S, lane, P, wl);
  }
}

// ---------------------------------------------------------------------------
// Scheduling (thread 0 of a worker)
// ---------------------------------------------------------------------------
__device__ void refresh_epoch(DeviceState& S, WorkerShared& ws) {
  unsigned long long e = ld_acquire(&S.published.v);
  if (e == ws.epoch) return;
  for (;;) {
    const Snapshot& sn = S.snap[e & 1];
    ws.n_lanes = ld_volatile(&sn.n_lanes);
    ws.lane = ld_volatile(&sn.map[worker_index()]);
    for (int l = 0; l < kMaxLanes; ++l) {
      ws.cap[l] = ld_volatile(&sn.cap[l]);
      ws.order[l] = ld_volatile(&sn.order[l]);
      ws.prog_gen[l] = ld_volatile(&sn.prog_gen[l]);
      ws.prog_snap[l] = ld_volatile(&sn.prog[l]);
    }
    const unsigned long long e2 = ld_acquire(&S.published.v);
    if (e2 == e) break;
    e = e2;
  }
  ws.epoch = e;
  if (S.mirror) {
    mirror_store(&S.mirror->worker_epoch_ns[worker_index()], globaltimer_ns());
    mirror_store(&S.mirror->worker_epoch[worker_index()], e);
  }
}

__device__ void start_idle_lanes(DeviceState& S, WorkerShared& ws) {
  for (uint32_t l = 0; l < ws.n_lanes && l < kMaxLanes; ++l) {
    ProgramDesc* P = ws.prog_snap[l];
    if (P == nullptr) continue;
    LaneState& L = S.lane[l];
    if (ws.prog_gen[l] == ld_acquire(&L.gen.v)) continue;
    if (ld_acquire(&L.running.v) != 0) continue;   // switches at its boundary
    if (atomicCAS(&L.running.v, 0u, 1u) != 0u) continue;
    mirror_lane(S, l, 2, 1);
    Worklist wl;
    wl.n = 0;
    begin_iteration(S, l, P, ws.prog_gen[l], true, wl);
    drain(S, l, P, wl);
  }
}

__device__ bool acquire_slot(LaneState& L, uint32_t cap) {
  unsigned int s = ld_volatile(&L.slots.v);
  while (s < cap) {
    const unsigned int old = atomicCAS(&L.slots.v, s, s + 1);
    if (old == s) return true;
    s = old;
  }
  return false;
}

__device__ void release_slot(DeviceState& S, WorkerShared& ws) {
  if (ws.slot_lane != kNoLane) {
    atomicSub(&S.lane[ws.slot_lane].slots.v, 1u);
    ws.slot_lane = kNoLane;
  }
}

__device__ __forceinline__ void load_desc(WorkerShared& ws, const ProgramDesc* P, uint32_t index) {
  const uint4* src = reinterpret_cast<const uint4*>(&P->tiles[index]);
  uint4* dst = reinterpret_cast<uint4*>(&ws.tile);
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[i] = __ldcg(src + i);
}

__device__ void load_tile(WorkerShared& ws, const ProgramDesc* P, uint32_t stage,
                          uint32_t index, uint32_t lane, uint32_t link, uint32_t n) {
  load_desc(ws, P, index);
  ws.prog = P;
  ws.stage = stage;
  ws.tile_lane = lane;
  ws.link_tile = link;
  ws.batch = n;
  ws.tile_left = n - 1;
  ws.tile_index = index;
}

__device__ bool try_lane(DeviceState& S, WorkerShared& ws, uint32_t l) {
  if (l >= kMaxLanes) return false;
  LaneState& L = S.lane[l];
  if (ld_acquire(&L.running.v) == 0) return false;
  const ProgramDesc* P = reinterpret_cast<const ProgramDesc*>(ld_acquire(&L.prog.v));
  if (P == nullptr) return false;
  uint32_t s, idx, n;
  if (ld_acquire(&L.link_tail.v) > ld_volatile(&L.link_head.v)) {
    bool held = ws.slot_lane == l;
    if (held && ld_volatile(&L.slots.v) > ws.cap[l]) {   // the grant shrank
      release_slot(S, ws);
      held = false;
    }
    if (!held) {
      release_slot(S, ws);
      if (acquire_slot(L, ws.cap[l])) {
        ws.slot_lane = l;
        held = true;
      }
    }
    if (held) {
      if (take_from(L, true, P, s, idx, n)) {
        load_tile(ws, P, s, idx, l, 1, n);
        return true;
      }
      release_slot(S, ws);
    }
  }
  if (take_from(L, false, P, s, idx, n)) {
    release_slot(S, ws);
    load_tile(ws, P, s, idx, l, 0, n);
    return true;
  }
  return false;
}

__device__ uint32_t schedule(DeviceState& S, WorkerShared& ws) {
  refresh_epoch(S, ws);
  if (ld_acquire(&S.shutdown.v) != 0) {
    release_slot(S, ws);
    return kCmdExit;
  }
  start_idle_lanes(S, ws);
  if (ws.lane < ws.n_lanes && try_lane(S, ws, ws.lane)) {
    ws.idle_ns = 64;
    return kCmdRun;
  }
  for (uint32_t i = 0; i < ws.n_lanes && i < kMaxLanes; ++i) {
    const uint32_t l = ws.order[i];
    if (l != ws.lane && try_lane(S, ws, l)) {
      ws.idle_ns = 64;
      return kCmdRun;
    }
  }
  release_slot(S, ws);
  __nanosleep(ws.idle_ns);
  ws.idle_ns = min(ws.idle_ns << 1, max(64u, ld_volatile(&S.idle_cap.v)));
  return kCmdIdle;
}

// Long waits inside tiles, for hang diagnosis (monofab::wait_note_address):
// per worker and warp, while one of its waits has lasted over kWaitNoteNs,
// site | parity << 8 | page << 16 and the device time the wait began.
__device__ unsigned long long g_wait_note[kMaxWorkers][kWorkerThreads / 32][2];
constexpr unsigned long long kWaitNoteNs = 50000000ull;

// Wait for the phase of parity `phase` of an mbarrier to complete, as
// cutlass's ClusterBarrier::wait does; a wait longer than kWaitNoteNs is
// noted once by the role's first thread.
__device__ __forceinline__ void wait_phase(uint64_t* bar, uint32_t phase, uint32_t site, uint32_t page) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  const bool leader = (threadIdx.x & 31) == 0;
  const int role = threadIdx.x >> 5;   // the warp
  unsigned long long t0 = 0;
  bool noted = false;
  for (uint32_t spins = 0;; ++spins) {
    uint32_t done;
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(done)
        : "r"(addr), "r"(phase)
        : "memory");
    if (done) {
      // the record names waits in progress only
      if (noted && leader) g_wait_note[worker_index()][role][0] = 0;
      return;
    }
    if (!noted) {
      // a poll may block for a while in try_wait: read the clock every time
      const unsigned long long now = globaltimer_ns();
      if (t0 == 0) {
        t0 = now;
      } else if (now - t0 > kWaitNoteNs) {
        noted = true;
        if (leader) {
          g_wait_note[worker_index()][role][1] = t0;
          g_wait_note[worker_index()][role][0] = site | (static_cast<unsigned long long>(phase) << 8) |
                                                 (static_cast<unsigned long long>(page) << 16);
        }
      }
    }
  }
}

__device__ __forceinline__ void note_tile(DeviceState& S, const WorkerShared& ws,
                                          unsigned long long state) {
  const uint32_t w = worker_index();
  S.worker_tile[w] = state | (static_cast<unsigned long long>(ws.tile_lane & 0x3f) << 56) |
                     (static_cast<unsigned long long>(ws.tile.kind & 0xfff) << 44) |
                     (static_cast<unsigned long long>(ws.stage & kStageMask) << 24) |
                     (ws.tile_index & 0xffffffull);
  if (state) S.worker_tile_ns[w] = globaltimer_ns();
}

// ---------------------------------------------------------------------------
// The persistent kernel body. Executor::run(tile, prog, arena, params, S)
// executes one tile with all threads of the worker.
// ---------------------------------------------------------------------------
template <class Executor>
__device__ __forceinline__ void worker_loop(DeviceState* S, const typename Executor::Params& params,
                            unsigned char* arena, uint32_t first_worker) {
  __shared__ WorkerShared ws;
  if (threadIdx.x == 0) {
    worker_slot() = first_worker + blockIdx.x;
    ws.epoch = ~0ull;
    ws.slot_lane = kNoLane;
    ws.idle_ns = 64;
    ws.lane = kNoLane;
    ws.n_lanes = 0;
    ws.more = 0;
  }
  __syncthreads();
  for (;;) {
    if (threadIdx.x == 0) ws.cmd = ws.more ? static_cast<uint32_t>(kCmdRun) : schedule(*S, ws);
    __syncthreads();
    const uint32_t cmd = ws.cmd;
    if (cmd == kCmdExit) break;
    if (cmd == kCmdRun) {
      if (threadIdx.x == 0) {
        if (ws.link_tile) {
          LaneState& L = S->lane[ws.tile_lane];
          const unsigned int c = atomicAdd(&L.inflight.v, 1u) + 1;
          atomicMax(&L.inflight_max.v, c);
        }
        note_tile(*S, ws, kWorkerRunning);
      }
      Executor::run(ws.tile, ws.prog, arena, params, *S);
      __syncthreads();
      if (threadIdx.x == 0) {
        if (ws.link_tile) atomicSub(&S->lane[ws.tile_lane].inflight.v, 1u);
        if (ws.tile_left > 0) {
          // the next tile of the claim, without scheduling
          --ws.tile_left;
          load_desc(ws, ws.prog, ++ws.tile_index);
          ws.more = 1;
        } else {
          // the whole worker's writes (ordered by the barriers above) become
          // visible device-wide before the claimed tiles count as done
          __threadfence();
          ws.more = 0;
          note_tile(*S, ws, kWorkerCompleting);
          complete_tile(*S, ws.tile_lane, ws.prog, ws.stage, ws.batch);
          note_tile(*S, ws, 0);
        }
      }
    }
    __syncthreads();
  }
}

}  // namespace monofab
