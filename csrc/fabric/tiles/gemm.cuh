// GEMM tiles: out = activations x weight^T for a 128-feature slice of the
// weight and a block of activation rows, on TMA and WGMMA.
//
// The weight slice is the WGMMA M operand (two consumer warpgroups of 64
// rows each) and the activation rows are the N operand, so a decode batch
// of a few tokens still maps onto 64xBM WGMMA instructions. Thread 0 of the
// producer warpgroup issues the TMA loads of each 64-deep K step into a
// four-stage ring of shared-memory buffers; the consumer warpgroups wait
// for each stage, run the WGMMAs, and release the stage. Weights reach the
// tile through a tensor map, so the same tile reads HBM (hot tier, staging
// buffer, dense weights) or pinned CPU DRAM over the link, depending only
// on which tensor map the tile names.
//
// Weights may be bf16 or FP8 (e4m3) with one scale per output feature. FP8
// weights cross the link and sit in HBM at one byte per weight; each
// consumer warpgroup converts its half of every K step to bf16 in shared
// memory (exactly: every e4m3 value is a bf16 value) and the epilogue
// applies the scales, so the activations stay bf16 (W8A16).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>

#include "fabric/model_types.h"
#include "fabric/types.h"

namespace monofab {
namespace gemm {

using namespace cute;
using bf16 = cute::bfloat16_t;

constexpr int kBN = 128;       // weight rows per tile (two warpgroups x 64)
constexpr int kBK = 64;        // K elements per stage
constexpr int kStages = 4;
constexpr int kConsumerBase = 128;   // consumer warpgroups: threads 128..383

__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }

// Four consecutive fp32 additions in one vector atomic (sm_90).
__device__ __forceinline__ void atomic_add4(float* p, float4 v) {
  atomicAdd(reinterpret_cast<float4*>(p), v);
}

// Four bf16 to and from floats, 8 bytes.
__device__ __forceinline__ float4 load_bf16x4(const bf16* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
  const float2 a = __bfloat1622float2(h[0]), b = __bfloat1622float2(h[1]);
  return make_float4(a.x, a.y, b.x, b.y);
}

__device__ __forceinline__ void store_bf16x4(bf16* p, float4 v) {
  uint2 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
  h[0] = __floats2bfloat162_rn(v.x, v.y);
  h[1] = __floats2bfloat162_rn(v.z, v.w);
  *reinterpret_cast<uint2*>(p) = u;
}

__device__ __forceinline__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

// 64 weight rows of one K step, FP8 as TMA loaded them (64 bytes a row),
// into the bf16 K-major 128-byte-swizzled layout WGMMA reads: 128 bytes a
// row, the 16-byte chunk j of row r at chunk j ^ (r % 8). t: 0..127.
__device__ __forceinline__ void fp8_rows_to_bf16(const uint8_t* src, uint8_t* dst, int t) {
  for (int i = t; i < 64 * 4; i += 128) {
    const int r = i >> 2, c = i & 3;
    const uint4 v = *reinterpret_cast<const uint4*>(src + r * 64 + c * 16);
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    uint32_t o[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      __nv_fp8x4_e4m3 q;
      q.__x = w[j];
      const float4 f = static_cast<float4>(q);
      const __nv_bfloat162 lo = __floats2bfloat162_rn(f.x, f.y);
      const __nv_bfloat162 hi = __floats2bfloat162_rn(f.z, f.w);
      o[2 * j] = *reinterpret_cast<const uint32_t*>(&lo);
      o[2 * j + 1] = *reinterpret_cast<const uint32_t*>(&hi);
    }
    uint8_t* row = dst + r * 128;
    const int sw = r & 7;
    *reinterpret_cast<uint4*>(row + (((2 * c) ^ sw) << 4)) = make_uint4(o[0], o[1], o[2], o[3]);
    *reinterpret_cast<uint4*>(row + (((2 * c + 1) ^ sw) << 4)) = make_uint4(o[4], o[5], o[6], o[7]);
  }
}

// ---------------------------------------------------------------------------
// Symmetric tile: a 128-row weight slice for up to BM activation rows, over
// all of K or, in a K-split GEMM, over the K range the tile names.
// ---------------------------------------------------------------------------
template <int BM, bool FP8>
struct Shape_ {
  // pipeline depth: as many K steps in flight as the arena holds, so a
  // tile streams its weights at close to the memory's rate
  static constexpr int kPipe = BM <= 32 ? 8 : BM <= 64 ? 6 : 4;
  using AtomLayout = GMMA::Layout_K_SW128_Atom<bf16>;
  // bf16 weights are read by WGMMA where TMA put them, one buffer a stage;
  // FP8 weights go through one converted buffer
  using SmemA = decltype(tile_to_shape(AtomLayout{}, make_shape(Int<kBN>{}, Int<kBK>{}, Int<FP8 ? 1 : kPipe>{})));
  using SmemB = decltype(tile_to_shape(AtomLayout{}, make_shape(Int<BM>{}, Int<kBK>{}, Int<kPipe>{})));
  using Atom = decltype(SM90::GMMA::ss_op_selector<bf16, bf16, float, Shape<_64, Int<BM>, Int<kBK>>>());
  using TiledMma = decltype(make_tiled_mma(Atom{}, Layout<Shape<_2, _1, _1>>{}));
  static constexpr int kBytesA = kBN * kBK * (FP8 ? 1 : 2);   // one stage of weights, as loaded
  static constexpr int kBytesB = BM * kBK * 2;
  static constexpr int kOffB = kPipe * kBytesA;
  static constexpr int kOffC = kOffB + kPipe * kBytesB;        // FP8: the converted slice
  static constexpr int kBytesC = FP8 ? kBN * kBK * 2 : 0;
  static constexpr int kOffX = kOffC + kBytesC;                // the epilogue stages the tile below here
  static constexpr int kOffBar = kOffX;
  static constexpr int kArena = kOffBar + 2 * kPipe * 8;
};

// Scale every accumulator by its weight row's scale (FP8 weights).
template <class Acc, class Coord>
__device__ __forceinline__ void scale_rows(Acc& acc, const Coord& tCcC, const float* scale, int row_lo,
                                           int row_hi) {
  CUTE_UNROLL
  for (int i = 0; i < size(acc); ++i) {
    const int r = get<0>(tCcC(i));
    acc(i) *= scale[r < 64 ? row_lo + r : row_hi + r - 64];
  }
}

template <int BM, bool FP8>
__device__ __forceinline__ void tile(const TileDesc& t, unsigned char* arena) {
  using S = Shape_<BM, FP8>;
  const GemmArgs& args = *reinterpret_cast<const GemmArgs*>(t.a[0]);
  const void* wmap = reinterpret_cast<const void*>(t.a[1]);
  const void* xmap = reinterpret_cast<const void*>(t.a[2]);
  const int n0 = static_cast<int>(t.i0);
  const int m0 = static_cast<int>(t.i1);
  const int m_valid = static_cast<int>(t.i2);
  const int wz = static_cast<int>(t.i3);
  // a K-split tile covers the K range in a[3] (first K tile | K tiles << 16)
  const int k_first = t.a[3] ? static_cast<int>(t.a[3] & 0xffffu) : 0;
  const int k_tiles = t.a[3] ? static_cast<int>((t.a[3] >> 16) & 0xffffu) : static_cast<int>(args.k_tiles);
  const uint32_t epi = args.epi;
  // rows of the two 64-row halves of the weight slice
  const int row_lo = n0;
  const int row_hi = epi == kEpiSiluMul ? static_cast<int>(args.up_offset) + n0 : n0 + 64;

  uint8_t* sA_load = reinterpret_cast<uint8_t*>(arena);
  bf16* sA = reinterpret_cast<bf16*>(FP8 ? arena + S::kOffC : arena);
  bf16* sB = reinterpret_cast<bf16*>(arena + S::kOffB);
  uint64_t* full = reinterpret_cast<uint64_t*>(arena + S::kOffBar);
  uint64_t* empty = full + S::kPipe;

  if (threadIdx.x == 0) {
    for (int s = 0; s < S::kPipe; ++s) {
      cutlass::arch::ClusterTransactionBarrier::init(&full[s], 1);
      cutlass::arch::ClusterBarrier::init(&empty[s], 2);
    }
    cutlass::arch::fence_barrier_init();
    cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(wmap));
    cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(xmap));
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    // producer: one thread issues every TMA load of the tile
    for (int k = 0; k < k_tiles; ++k) {
      const int s = k % S::kPipe;
      const uint32_t ph = (k / S::kPipe) & 1;
      cutlass::arch::ClusterBarrier::wait(&empty[s], ph ^ 1);
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&full[s], S::kBytesA + S::kBytesB);
      uint8_t* a = sA_load + s * S::kBytesA;
      const int kk = (k_first + k) * kBK;
      SM90_TMA_LOAD_3D::copy(wmap, &full[s], 0, a, kk, row_lo, wz);
      SM90_TMA_LOAD_3D::copy(wmap, &full[s], 0, a + S::kBytesA / 2, kk, row_hi, wz);
      SM90_TMA_LOAD_2D::copy(xmap, &full[s], 0, sB + s * (BM * kBK), kk, m0);
    }
    return;
  }
  if (threadIdx.x < kConsumerBase) return;

  // consumers
  const int tc = threadIdx.x - kConsumerBase;
  const int wg = tc / 128;   // 0: rows 0..63, 1: rows 64..127
  typename S::TiledMma tm;
  auto thr = tm.get_thread_slice(tc);
  Tensor tSA = make_tensor(make_smem_ptr(sA), typename S::SmemA{});
  Tensor tSB = make_tensor(make_smem_ptr(sB), typename S::SmemB{});
  Tensor tCsA = thr.partition_A(tSA);
  Tensor tCsB = thr.partition_B(tSB);
  Tensor tCrA = thr.make_fragment_A(tCsA);
  Tensor tCrB = thr.make_fragment_B(tCsB);
  Tensor acc = partition_fragment_C(tm, Shape<Int<kBN>, Int<BM>>{});
  clear(acc);

  for (int k = 0; k < k_tiles; ++k) {
    const int s = k % S::kPipe;
    const uint32_t ph = (k / S::kPipe) & 1;
    cutlass::arch::ClusterTransactionBarrier::wait(&full[s], ph);
    if constexpr (FP8) {
      // each warpgroup converts the half of the slice its WGMMAs read; the
      // previous K step's WGMMAs have completed (wait<0> below)
      fp8_rows_to_bf16(sA_load + s * S::kBytesA + wg * (S::kBytesA / 2),
                       reinterpret_cast<uint8_t*>(sA) + wg * 64 * 128, tc & 127);
      fence_async_shared();
      cutlass::arch::NamedBarrier::sync(128, 2 + wg);
    }
    warpgroup_fence_operand(acc);
    warpgroup_arrive();
    cute::gemm(tm, tCrA(_, _, _, FP8 ? 0 : s), tCrB(_, _, _, s), acc);
    warpgroup_commit_batch();
    if constexpr (FP8) {
      // the one converted buffer is rewritten next step: drain now
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc);
      if ((tc & 127) == 0) cutlass::arch::ClusterBarrier::arrive(&empty[s]);
    } else {
      // this step's WGMMAs stay in flight; the previous step's are done, so
      // its stage goes back to the producer
      warpgroup_wait<1>();
      warpgroup_fence_operand(acc);
      if (k > 0 && (tc & 127) == 0)
        cutlass::arch::ClusterBarrier::arrive(&empty[(k - 1) % S::kPipe]);
    }
  }
  if constexpr (!FP8) {
    warpgroup_wait<0>();
    warpgroup_fence_operand(acc);
  }

  // epilogue: (row, col) of every accumulator element; row is the weight
  // feature within the slice, col the activation row within the block
  Tensor cC = make_identity_tensor(Shape<Int<kBN>, Int<BM>>{});
  Tensor tCcC = thr.partition_C(cC);
  if constexpr (FP8) scale_rows(acc, tCcC, reinterpret_cast<const float*>(t.a[4]), row_lo, row_hi);

  // The tile goes through shared memory (the pipeline's buffers, all
  // consumed), [activation row][feature], and leaves along the features:
  // each warp writes one activation row per pass in 16-byte or 8-byte
  // pieces, and the weighted add uses 4-wide vector atomics.
  constexpr int ldc = kBN + 4;
  static_assert(BM * ldc * 4 <= S::kOffX, "the staged tile must fit the pipeline buffers");
  float* cs = reinterpret_cast<float*>(arena);
  // the staged tile covers buffers the other warpgroup's last WGMMAs may
  // still be reading (they drift apart when weights arrive over the link):
  // both warpgroups finish first
  cutlass::arch::NamedBarrier::sync(256, 1);
  CUTE_UNROLL
  for (int i = 0; i < size(acc); ++i) cs[get<1>(tCcC(i)) * ldc + get<0>(tCcC(i))] = acc(i);
  cutlass::arch::NamedBarrier::sync(256, 1);
  const int warp = tc >> 5, lane = tc & 31;
  const int n_limit = static_cast<int>(args.n_limit);

  if (epi == kEpiSiluMul) {
    // features 0..63 are the gate, 64..127 the up projection: 64 outputs a
    // row, four per lane, two rows per warp and pass
    bf16* out = reinterpret_cast<bf16*>(args.out);
    const int j = 4 * (lane & 15);
    for (int c = 2 * warp + (lane >> 4); c < m_valid; c += 16) {
      const float* g = cs + c * ldc + j;
      const float* u = g + 64;
      bf16* o = out + static_cast<size_t>(m0 + c) * args.ld_out + n0 + j;
      if (n0 + j + 4 <= n_limit) {
        store_bf16x4(o, make_float4(silu(g[0]) * u[0], silu(g[1]) * u[1], silu(g[2]) * u[2],
                                    silu(g[3]) * u[3]));
      } else {
        for (int q = 0; q < 4 && n0 + j + q < n_limit; ++q) o[q] = bf16(silu(g[q]) * u[q]);
      }
    }
    // the next tile's TMA loads overwrite these buffers
    fence_async_shared();
    return;
  }

  const int j = 4 * lane;   // 128 features a row, four per lane
  const int n = n0 + j;
  for (int c = warp; c < m_valid; c += 8) {
    if (n >= n_limit) continue;
    const float* src = cs + c * ldc + j;
    float4 v = make_float4(src[0], src[1], src[2], src[3]);
    const size_t m = static_cast<size_t>(m0 + c);
    const bool whole = n + 4 <= n_limit;
    if (epi == kEpiStoreBf16) {
      bf16* o = reinterpret_cast<bf16*>(args.out) + m * args.ld_out + n;
      const bf16* bias = args.bias ? reinterpret_cast<const bf16*>(args.bias) + n : nullptr;
      const bf16* res = args.residual ? reinterpret_cast<const bf16*>(args.residual) + m * args.ld_res + n
                                      : nullptr;
      if (whole) {
        if (bias) {
          const float4 b = load_bf16x4(bias);
          v.x += b.x; v.y += b.y; v.z += b.z; v.w += b.w;
        }
        if (res) {
          const float4 r = load_bf16x4(res);
          v.x += r.x; v.y += r.y; v.z += r.z; v.w += r.w;
        }
        store_bf16x4(o, v);
      } else {
        const float vs[4] = {v.x, v.y, v.z, v.w};
        for (int q = 0; q < 4 && n + q < n_limit; ++q) {
          float x = vs[q];
          if (bias) x += float(bias[q]);
          if (res) x += float(res[q]);
          o[q] = bf16(x);
        }
      }
    } else if (epi == kEpiWeightedAdd) {
      const int tok = reinterpret_cast<const int*>(args.pair_token)[m];
      const float w = reinterpret_cast<const float*>(args.pair_weight)[m];
      float* o = reinterpret_cast<float*>(args.out) + static_cast<size_t>(tok) * args.ld_out + n;
      if (whole) {
        atomic_add4(o, make_float4(w * v.x, w * v.y, w * v.z, w * v.w));
      } else {
        const float vs[4] = {v.x, v.y, v.z, v.w};
        for (int q = 0; q < 4 && n + q < n_limit; ++q) atomicAdd(o + q, w * vs[q]);
      }
    } else if (epi == kEpiAtomicF32) {
      // partial sums of a K range; the first range adds the bias
      float* o = reinterpret_cast<float*>(args.out) + m * args.ld_out + n;
      const bf16* bias = k_first == 0 && args.bias ? reinterpret_cast<const bf16*>(args.bias) + n : nullptr;
      if (whole) {
        if (bias) {
          const float4 b = load_bf16x4(bias);
          v.x += b.x; v.y += b.y; v.z += b.z; v.w += b.w;
        }
        atomic_add4(o, v);
      } else {
        const float vs[4] = {v.x, v.y, v.z, v.w};
        for (int q = 0; q < 4 && n + q < n_limit; ++q) atomicAdd(o + q, vs[q] + (bias ? float(bias[q]) : 0.f));
      }
    } else {
      float* o = reinterpret_cast<float*>(args.out) + m * args.ld_out + n;
      if (whole) {
        *reinterpret_cast<float4*>(o) = v;
      } else {
        const float vs[4] = {v.x, v.y, v.z, v.w};
        for (int q = 0; q < 4 && n + q < n_limit; ++q) o[q] = vs[q];
      }
    }
  }
  // the next tile's TMA loads overwrite these buffers
  fence_async_shared();
}

template <bool FP8>
__device__ __forceinline__ void run_form(const TileDesc& t, unsigned char* arena, uint32_t bm) {
  switch (bm) {
    case 16: tile<16, FP8>(t, arena); break;
    case 32: tile<32, FP8>(t, arena); break;
    case 64: tile<64, FP8>(t, arena); break;
    default: tile<128, FP8>(t, arena); break;
  }
}

__device__ __forceinline__ void run(const TileDesc& t, unsigned char* arena) {
  const GemmArgs& a = *reinterpret_cast<const GemmArgs*>(t.a[0]);
  if (a.w_fp8) run_form<true>(t, arena, a.bm);
  else run_form<false>(t, arena, a.bm);
}

// ---------------------------------------------------------------------------
// Asymmetric tile: the DRAM-expert form that reads every weight byte once.
// It takes one K range (at most kAsymMaxKT K tiles) of a 128-row weight
// slice, keeps it in shared memory as bf16, and walks all rows routed to the
// expert in blocks of BM, so the slice crosses the link once whatever the
// rows. FP8 slices arrive through a two-slot ring and are converted into
// the resident slice. Partial sums leave through atomics: kEpiAtomicF32 into
// the w13 workspace, which a reduce tile turns into silu(gate) * up, and
// kEpiWeightedAdd for w2, whose epilogue is linear, straight into the
// tokens' accumulators.
// ---------------------------------------------------------------------------
template <int BM, bool FP8>
struct AsymShape {
  using AtomLayout = GMMA::Layout_K_SW128_Atom<bf16>;
  using SmemA = decltype(tile_to_shape(AtomLayout{}, make_shape(Int<kBN>{}, Int<kBK>{}, Int<kAsymMaxKT>{})));
  using SmemB = decltype(tile_to_shape(AtomLayout{}, make_shape(Int<BM>{}, Int<kBK>{}, Int<kStages>{})));
  using Atom = decltype(SM90::GMMA::ss_op_selector<bf16, bf16, float, Shape<_64, Int<BM>, Int<kBK>>>());
  using TiledMma = decltype(make_tiled_mma(Atom{}, Layout<Shape<_2, _1, _1>>{}));
  static constexpr int kBytesA = kBN * kBK * 2;   // one resident bf16 K tile
  static constexpr int kBytesA8 = kBN * kBK;      // one FP8 K tile as loaded
  static constexpr int kSlots8 = FP8 ? 2 : 0;
  static constexpr int kBytesB = BM * kBK * 2;
  static constexpr int kOffA8 = kAsymMaxKT * kBytesA;
  static constexpr int kOffB = kOffA8 + kSlots8 * kBytesA8;
  static constexpr int kOffBar = kOffB + kStages * kBytesB;
  static constexpr int kArena = kOffBar + (kAsymMaxKT + 2 * kStages + 4) * 8;
};

template <int BM, bool FP8>
__device__ __forceinline__ void asym_tile(const TileDesc& t, unsigned char* arena) {
  using S = AsymShape<BM, FP8>;
  const GemmArgs& args = *reinterpret_cast<const GemmArgs*>(t.a[0]);
  const void* wmap = reinterpret_cast<const void*>(t.a[1]);
  const void* xmap = reinterpret_cast<const void*>(t.a[2]);
  const int n0 = static_cast<int>(t.i0);
  const int m_first = static_cast<int>(t.i1);
  const int m_rows = static_cast<int>(t.i2);
  const int wz = static_cast<int>(t.i3);
  const int k0 = static_cast<int>(t.a[3] & 0xffffu);
  const int kt = static_cast<int>((t.a[3] >> 16) & 0xffffu);
  const uint32_t epi = args.epi;
  const int row_lo = n0;
  const int row_hi = args.up_offset ? static_cast<int>(args.up_offset) + n0 : n0 + 64;
  const int blocks = (m_rows + BM - 1) / BM;

  bf16* sA = reinterpret_cast<bf16*>(arena);
  uint8_t* sA8 = reinterpret_cast<uint8_t*>(arena + S::kOffA8);
  bf16* sB = reinterpret_cast<bf16*>(arena + S::kOffB);
  uint64_t* wfull = reinterpret_cast<uint64_t*>(arena + S::kOffBar);
  uint64_t* full = wfull + kAsymMaxKT;
  uint64_t* empty = full + kStages;
  uint64_t* full8 = empty + kStages;
  uint64_t* empty8 = full8 + 2;

  if (threadIdx.x == 0) {
    for (int k = 0; k < kAsymMaxKT; ++k) cutlass::arch::ClusterTransactionBarrier::init(&wfull[k], 1);
    for (int s = 0; s < kStages; ++s) {
      cutlass::arch::ClusterTransactionBarrier::init(&full[s], 1);
      cutlass::arch::ClusterBarrier::init(&empty[s], 2);
    }
    for (int s = 0; s < 2; ++s) {
      cutlass::arch::ClusterTransactionBarrier::init(&full8[s], 1);
      cutlass::arch::ClusterBarrier::init(&empty8[s], 2);
    }
    cutlass::arch::fence_barrier_init();
    cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(wmap));
    cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(xmap));
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    // producer. bf16: the whole slice first, one barrier per K tile, so the
    // first row block starts on the first tile that lands. FP8: each K tile
    // goes through the ring just ahead of the activations of the first row
    // block that need it.
    if constexpr (!FP8) {
      for (int k = 0; k < kt; ++k) {
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&wfull[k], S::kBytesA);
        bf16* a = sA + k * (kBN * kBK);
        SM90_TMA_LOAD_3D::copy(wmap, &wfull[k], 0, a, (k0 + k) * kBK, row_lo, wz);
        SM90_TMA_LOAD_3D::copy(wmap, &wfull[k], 0, a + 64 * kBK, (k0 + k) * kBK, row_hi, wz);
      }
    }
    int it = 0;
    for (int b = 0; b < blocks; ++b)
      for (int k = 0; k < kt; ++k, ++it) {
        if constexpr (FP8) {
          if (b == 0) {
            const int slot = k & 1;
            cutlass::arch::ClusterBarrier::wait(&empty8[slot], ((k >> 1) & 1) ^ 1);
            cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&full8[slot], S::kBytesA8);
            uint8_t* a = sA8 + slot * S::kBytesA8;
            SM90_TMA_LOAD_3D::copy(wmap, &full8[slot], 0, a, (k0 + k) * kBK, row_lo, wz);
            SM90_TMA_LOAD_3D::copy(wmap, &full8[slot], 0, a + S::kBytesA8 / 2, (k0 + k) * kBK, row_hi, wz);
          }
        }
        const int s = it % kStages;
        const uint32_t ph = (it / kStages) & 1;
        cutlass::arch::ClusterBarrier::wait(&empty[s], ph ^ 1);
        cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&full[s], S::kBytesB);
        SM90_TMA_LOAD_2D::copy(xmap, &full[s], 0, sB + s * (BM * kBK), (k0 + k) * kBK, m_first + b * BM);
      }
    return;
  }
  if (threadIdx.x < kConsumerBase) return;

  const int tc = threadIdx.x - kConsumerBase;
  const int wg = tc / 128;
  typename S::TiledMma tm;
  auto thr = tm.get_thread_slice(tc);
  Tensor tSA = make_tensor(make_smem_ptr(sA), typename S::SmemA{});
  Tensor tSB = make_tensor(make_smem_ptr(sB), typename S::SmemB{});
  Tensor tCsA = thr.partition_A(tSA);
  Tensor tCsB = thr.partition_B(tSB);
  Tensor tCrA = thr.make_fragment_A(tCsA);
  Tensor tCrB = thr.make_fragment_B(tCsB);
  Tensor acc = partition_fragment_C(tm, Shape<Int<kBN>, Int<BM>>{});
  Tensor cC = make_identity_tensor(Shape<Int<kBN>, Int<BM>>{});
  Tensor tCcC = thr.partition_C(cC);
  const float* scale = reinterpret_cast<const float*>(t.a[4]);

  int it = 0;
  for (int b = 0; b < blocks; ++b) {
    clear(acc);
    for (int k = 0; k < kt; ++k, ++it) {
      const int s = it % kStages;
      const uint32_t ph = (it / kStages) & 1;
      if (b == 0) {
        if constexpr (FP8) {
          const int slot = k & 1;
          cutlass::arch::ClusterTransactionBarrier::wait(&full8[slot], (k >> 1) & 1);
          fp8_rows_to_bf16(sA8 + slot * S::kBytesA8 + wg * (S::kBytesA8 / 2),
                           reinterpret_cast<uint8_t*>(sA + k * (kBN * kBK)) + wg * 64 * 128, tc & 127);
          fence_async_shared();
          cutlass::arch::NamedBarrier::sync(128, 2 + wg);
          if ((tc & 127) == 0) cutlass::arch::ClusterBarrier::arrive(&empty8[slot]);
        } else {
          cutlass::arch::ClusterTransactionBarrier::wait(&wfull[k], 0);
        }
      }
      cutlass::arch::ClusterTransactionBarrier::wait(&full[s], ph);
      warpgroup_fence_operand(acc);
      warpgroup_arrive();
      cute::gemm(tm, tCrA(_, _, _, k), tCrB(_, _, _, s), acc);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(acc);
      if ((tc & 127) == 0) cutlass::arch::ClusterBarrier::arrive(&empty[s]);
    }
    if constexpr (FP8) scale_rows(acc, tCcC, scale, row_lo, row_hi);
    const int base = m_first + b * BM;
    const int valid = min(BM, m_rows - b * BM);
    CUTE_UNROLL
    for (int i = 0; i < size(acc); ++i) {
      const int r = get<0>(tCcC(i));
      const int c = get<1>(tCcC(i));
      if (c >= valid) continue;
      const size_t m = static_cast<size_t>(base + c);
      if (epi == kEpiAtomicF32) {
        const int wrow = r < 64 ? row_lo + r : row_hi + (r - 64);
        atomicAdd(reinterpret_cast<float*>(args.out) + m * args.ld_out + wrow, acc(i));
      } else {   // kEpiWeightedAdd
        const int n = n0 + r;
        if (n >= static_cast<int>(args.n_limit)) continue;
        const int tok = reinterpret_cast<const int*>(args.pair_token)[m];
        const float w = reinterpret_cast<const float*>(args.pair_weight)[m];
        atomicAdd(reinterpret_cast<float*>(args.out) + static_cast<size_t>(tok) * args.ld_out + n, w * acc(i));
      }
    }
  }
}

template <bool FP8>
__device__ __forceinline__ void run_asym_form(const TileDesc& t, unsigned char* arena, uint32_t bm) {
  switch (bm) {
    case 16: asym_tile<16, FP8>(t, arena); break;
    case 32: asym_tile<32, FP8>(t, arena); break;
    default: asym_tile<64, FP8>(t, arena); break;
  }
}

__device__ __forceinline__ void run_asym(const TileDesc& t, unsigned char* arena) {
  const GemmArgs& a = *reinterpret_cast<const GemmArgs*>(t.a[0]);
  if (a.w_fp8) run_asym_form<true>(t, arena, a.bm);
  else run_asym_form<false>(t, arena, a.bm);
}

template <int A, int B>
constexpr int cmax() { return A > B ? A : B; }
// shared memory the largest GEMM tile needs
template <bool FP8>
constexpr int sym_arena() {
  return cmax<cmax<Shape_<16, FP8>::kArena, Shape_<32, FP8>::kArena>(),
              cmax<Shape_<64, FP8>::kArena, Shape_<128, FP8>::kArena>()>();
}
constexpr int kArenaBytes = cmax<cmax<sym_arena<false>(), sym_arena<true>()>(),
                                 cmax<AsymShape<64, false>::kArena, AsymShape<64, true>::kArena>()>();
static_assert(kArenaBytes <= 200 * 1024, "the GEMM tiles must fit the fabric's 200 KB arena");

}  // namespace gemm
}  // namespace monofab
