// Data layouts shared by the host runtime and the MonoFab kernel.
//
// A lane runs a program: a DAG of stages, each a set of tiles that become
// ready together. The kernel's workers pull tiles from the stages that are
// active on their lane. DRAM-expert tiles (tiles whose weights resolve to
// CPU DRAM and therefore read over the link) sit in stages flagged
// kStageLink and are taken only under one of the lane's link slots.
#pragma once
#include <cstdint>

namespace monofab {

constexpr int kMaxLanes = 4;
constexpr int kMaxWorkers = 256;
constexpr uint32_t kRingCap = 1u << 14;      // stage ids per ring
constexpr uint32_t kNoLane = 0xffu;
constexpr int kWorkerThreads = 384;          // 1 producer + 2 consumer warpgroups

enum TileKind : uint16_t {
  kTileNop = 0,
  kTileSpin = 1,          // test: spin for a[0] ns
  kTileRead = 2,          // stream a[1] bytes from a[0]
  kTileProbe = 3,         // calibration: dependent-load chase
  kTileZeroRuntime = 4,   // reset stage bookkeeping between iterations
  kFirstModelTile = 32,   // model tiles are numbered from here
};

enum StageFlags : uint32_t {
  kStageLink = 1u << 0,   // DRAM-expert tiles, gated by link slots
  kStageReset = 1u << 1,  // completion ends one iteration of the program
  kStageMark = 1u << 2,   // completion is reported to the host (mark value)
  kStageSkip = 1u << 3,   // out of this iteration (an unrouted expert's chain): passed over
};

// 64-byte tile descriptor. Model tiles interpret i* and a* per kind.
struct alignas(64) TileDesc {
  uint16_t kind;
  uint8_t lane;
  uint8_t flags;
  uint32_t stage;
  uint32_t i0, i1, i2, i3;
  uint64_t a[5];
};
static_assert(sizeof(TileDesc) == 64, "TileDesc must be 64 bytes");

struct alignas(32) StageStatic {
  uint32_t first;       // first tile of the stage in the program's tile array
  uint32_t count;       // number of tiles; dynamic stages set it at run time
  uint32_t succ_first;  // successors: [succ_first, succ_first + succ_count)
  uint32_t succ_count;
  uint32_t deps_init;   // number of predecessor stages
  uint32_t flags;       // StageFlags
  uint32_t mark;        // reported to the host when kStageMark is set
  uint32_t pad;
};
static_assert(sizeof(StageStatic) == 32, "StageStatic must be 32 bytes");

// Per-stage bookkeeping. tag_next packs the iteration tag of the current
// activation with the index of the next tile to hand out, so a claim
// validates the tag and takes the tile in one compare-and-swap.
struct alignas(16) StageRuntime {
  unsigned long long tag_next;
  unsigned int done;
  unsigned int deps;
};
static_assert(sizeof(StageRuntime) == 16, "StageRuntime must be 16 bytes");

// One uploaded program; lives in device memory and never changes after
// upload, except the counts of dynamic stages and the runtime array.
struct alignas(64) ProgramDesc {
  TileDesc* tiles;
  StageStatic* stages;
  StageRuntime* runtime;
  uint32_t* succ;
  uint32_t n_tiles;
  uint32_t n_stages;
  uint32_t first_stage;   // activated at the start of every iteration
  uint32_t reset_stage;   // its completion ends an iteration
  uint32_t iterations;    // 0: repeat until a newer program replaces it
  uint32_t gen;           // unique per upload
};

// The plan the host publishes as an epoch. Double-buffered in device memory.
struct alignas(128) Snapshot {
  unsigned long long epoch;
  uint32_t n_workers;
  uint32_t n_lanes;
  uint32_t cap[kMaxLanes];        // link slots per lane
  uint32_t order[kMaxLanes];      // lanes in borrowing order
  uint32_t prog_gen[kMaxLanes];   // generation of the program to run
  uint32_t form[kMaxLanes];       // DRAM-expert kernel form: symmetric tile height (16/32/64/128)
  ProgramDesc* prog[kMaxLanes];   // program per lane, or null
  uint8_t map[kMaxWorkers];       // worker -> lane (kNoLane: none)
};

// Written by the device into pinned host memory; read by the host loop.
struct alignas(64) LaneMirror {
  unsigned long long iterations;  // completed iterations since start
  unsigned long long gen;         // generation of the running program
  unsigned long long running;
  unsigned long long mark;        // last completed marked stage
  unsigned long long mark_ns;     // device global time of that completion
  unsigned long long iter_ns;     // device global time at which the last iteration ended
  unsigned long long pad[2];
};

struct alignas(64) HostMirror {
  LaneMirror lane[kMaxLanes];
  unsigned long long worker_epoch[kMaxWorkers];
  unsigned long long worker_epoch_ns[kMaxWorkers];
};

}  // namespace monofab
