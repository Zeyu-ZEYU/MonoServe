// Attention tiles on ThunderKittens' Hopper building blocks: warpgroup
// WGMMAs on swizzled shared tiles and register-tile online softmax, with
// K/V pages loaded by TMA straight from the paged KV cache.
//
// Thread 0 of the producer warpgroup loads the query tiles and streams the
// KV pages of the tile's chunk through a three-stage ring; the two
// consumer warpgroups each own 64 query rows.
#pragma once
#include "kittens.cuh"

#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>

#include "fabric/model_types.h"
#include "fabric/runtime.cuh"
#include "fabric/types.h"

namespace monofab {
namespace attn {

namespace tk = kittens;
constexpr int kD = 128;
constexpr int kRows = 64;   // query rows per consumer warpgroup, keys per page
constexpr int kStages = 3;
constexpr int kConsumerWarps = 8;   // two consumer warpgroups
using q_st = tk::st_bf<kRows, kD>;
using kv_st = tk::st_bf<kRows, kD>;
using o_st = tk::st_fl<kRows, kD>;
using att_rt = tk::rt_fl<16, kRows>;
using att_bf = tk::rt_bf<16, kRows>;
using o_rt = tk::rt_fl<16, kD>;
using vec_rt = tk::col_vec<att_rt>;
using vec_sv = tk::sv_fl<kRows>;

constexpr int kOffQ = 0;
constexpr int kOffK = kOffQ + 2 * static_cast<int>(sizeof(q_st));
constexpr int kOffV = kOffK + kStages * static_cast<int>(sizeof(kv_st));
constexpr int kOffO = kOffV + kStages * static_cast<int>(sizeof(kv_st));
constexpr int kOffVec = kOffO + 2 * static_cast<int>(sizeof(o_st));
constexpr int kOffBar = kOffVec + 4 * static_cast<int>(sizeof(vec_sv));
constexpr int kArenaBytes = kOffBar + 16 * 8;

constexpr float kNegInf = -1e30f;

struct Smem {
  q_st* q;       // [2]
  kv_st* k;      // [kStages]
  kv_st* v;      // [kStages]
  o_st* o;       // [2] epilogue staging
  vec_sv* vec;   // [4] m/l exchange
  uint64_t* bar; // q, k[3], v[3], done[3]
};

__device__ __forceinline__ Smem carve(unsigned char* arena) {
  Smem s;
  s.q = reinterpret_cast<q_st*>(arena + kOffQ);
  s.k = reinterpret_cast<kv_st*>(arena + kOffK);
  s.v = reinterpret_cast<kv_st*>(arena + kOffV);
  s.o = reinterpret_cast<o_st*>(arena + kOffO);
  s.vec = reinterpret_cast<vec_sv*>(arena + kOffVec);
  s.bar = reinterpret_cast<uint64_t*>(arena + kOffBar);
  return s;
}

// Element (row, col) of a warp's 16-row register tile: data[0..3] of each
// 16x16 subtile hold (g, c), (g + 8, c), (g, c + 8), (g + 8, c + 8) with
// g = lane / 4 and c = 16 j + 2 (lane % 4).
template <class RT, class F>
__device__ __forceinline__ void for_each(RT& t, F f) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int c0 = 2 * (lane & 3);
#pragma unroll
  for (int j = 0; j < RT::width; ++j) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int r = g + ((k & 1) ? 8 : 0);
      const int c = 16 * j + c0 + ((k & 2) ? 8 : 0);
      f(t.tiles[0][j].data[k].x, r, c);
      f(t.tiles[0][j].data[k].y, r, c + 1);
    }
  }
}

// Set scores of keys the query rows may not see to -inf.
__device__ __forceinline__ void mask_scores(att_rt& s, int warp_row0, int qpos0, int key0,
                                            int kv_len) {
  for_each(s, [&](float& v, int r, int c) {
    const int qp = qpos0 + warp_row0 + r;
    const int kp = key0 + c;
    if (kp > qp || kp >= kv_len) v = kNegInf;
  });
}

struct Softmax {
  vec_rt m, l, m_prev;
  o_rt o;
};

// One KV page: s = q k^T, online softmax update, o += p v.
__device__ __forceinline__ void consume_page(Softmax& st, q_st& q, kv_st& k, kv_st& v,
                                             uint64_t* k_bar, uint64_t* v_bar, uint32_t phase,
                                             float scale_log2, bool masked, int warp_row0,
                                             int qpos0, int key0, int kv_len, int page) {
  att_rt s;
  att_bf p;
  wait_phase(k_bar, phase, 3, page);
  tk::warpgroup::mm_ABt(s, q, k);
  tk::warpgroup::mma_async_wait();
  if (masked) mask_scores(s, warp_row0, qpos0, key0, kv_len);
  tk::warp::copy(st.m_prev, st.m);
  tk::warp::row_max(st.m, s, st.m);
  tk::warp::mul(s, s, scale_log2);
  vec_rt ms;
  tk::warp::mul(ms, st.m, scale_log2);
  tk::warp::sub_row(s, s, ms);
  tk::warp::exp2(s, s);
  tk::warp::mul(st.m_prev, st.m_prev, scale_log2);
  tk::warp::sub(st.m_prev, st.m_prev, ms);
  tk::warp::exp2(st.m_prev, st.m_prev);     // rescale factor of the old state
  tk::warp::mul(st.l, st.l, st.m_prev);
  tk::warp::row_sum(st.l, s, st.l);
  tk::warp::mul_row(st.o, st.o, st.m_prev);
  tk::warp::copy(p, s);
  wait_phase(v_bar, phase, 4, page);
  tk::warpgroup::mma_AB(st.o, p, v);
  tk::warpgroup::mma_async_wait();
}

__device__ __forceinline__ void init_barriers(Smem& sm) {
  if (threadIdx.x == 0) {
    cutlass::arch::ClusterTransactionBarrier::init(&sm.bar[0], 1);
    for (int s = 0; s < kStages; ++s) {
      cutlass::arch::ClusterTransactionBarrier::init(&sm.bar[1 + s], 1);
      cutlass::arch::ClusterTransactionBarrier::init(&sm.bar[4 + s], 1);
      cutlass::arch::ClusterBarrier::init(&sm.bar[7 + s], kConsumerWarps);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
}

// Producer: stream pages [p0, p1) of request r through the ring.
__device__ __forceinline__ void produce_pages(const AttnArgs& A, Smem& sm, int r, int kvh, int p0,
                                              int p1) {
  const int* bt = reinterpret_cast<const int*>(A.block_table) + static_cast<size_t>(r) * A.bt_stride;
  for (int j = p0, it = 0; j < p1; ++j, ++it) {
    const int s = it % kStages;
    const uint32_t ph = (it / kStages) & 1;
    wait_phase(&sm.bar[7 + s], ph ^ 1, 1, it);
    const int block = bt[j];
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&sm.bar[1 + s], sizeof(kv_st));
    cute::SM90_TMA_LOAD_5D::copy(reinterpret_cast<const void*>(A.k_map), &sm.bar[1 + s], 0,
                                 &sm.k[s], 0, 0, 0, kvh, block);
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&sm.bar[4 + s], sizeof(kv_st));
    cute::SM90_TMA_LOAD_5D::copy(reinterpret_cast<const void*>(A.v_map), &sm.bar[4 + s], 0,
                                 &sm.v[s], 0, 0, 0, kvh, block);
  }
}

// Every consumer warp releases a page's slot once it is done with the page,
// and the producer refills the slot only after all of them have. Released
// once per warpgroup, a slot could be refilled while one warp of the group
// still had to wait for a page it only skips; the barrier would then be two
// phases ahead of that warp, whose parity wait would take the next refill
// for its own and deadlock the tile.
__device__ __forceinline__ void release_page(Smem& sm, int it) {
  if ((threadIdx.x & 31) == 0) cutlass::arch::ClusterBarrier::arrive(&sm.bar[7 + it % kStages]);
}

// ---------------------------------------------------------------------------
// Prefill
// ---------------------------------------------------------------------------
__device__ __forceinline__ void prefill(const TileDesc& t, unsigned char* arena) {
  const AttnArgs& A = *reinterpret_cast<const AttnArgs*>(t.a[0]);
  const int r = t.i0, h = t.i1, qb = t.i2, c = t.i3;
  const int* info = reinterpret_cast<const int*>(A.req_info) + 4 * r;
  const int q_row = info[0], pos0 = info[1], n_rows = info[2], kv_len = info[3];
  const int kvh = h / A.group;
  const int qr0 = qb * 128;                        // first query row of the tile
  const int last_q = min(n_rows, qr0 + 128) - 1;   // last valid row
  const int keys_end = min(kv_len, pos0 + last_q + 1);
  const int chunk_keys = A.chunk_blocks * kRows;
  const int k0 = c * chunk_keys;
  const int k1 = min(keys_end, k0 + chunk_keys);
  const int p0 = k0 / kRows, p1 = (k1 + kRows - 1) / kRows;
  const bool first = c == 0;
  const bool last = k0 + chunk_keys >= keys_end;
  Smem sm = carve(arena);
  init_barriers(sm);

  if (threadIdx.x == 0) {
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&sm.bar[0], 2 * sizeof(q_st));
    for (int w = 0; w < 2; ++w)
      cute::SM90_TMA_LOAD_5D::copy(reinterpret_cast<const void*>(A.q_map), &sm.bar[0], 0,
                                   &sm.q[w], 0, q_row + qr0 + 64 * w, 0, h, 0);
    produce_pages(A, sm, r, kvh, p0, p1);
    return;
  }
  if (threadIdx.x < 128) return;

  const int wg = (threadIdx.x - 128) / 128;            // 0 or 1
  const int wrow0 = 16 * (tk::warpid() % 4);           // warp's rows within the WG tile
  const int qrow_wg = qr0 + 64 * wg;                   // first query row of this WG
  const int qpos_wg = pos0 + qrow_wg;
  Softmax st;
  const size_t row_base = static_cast<size_t>(q_row + qrow_wg);
  float* ws_o = reinterpret_cast<float*>(A.ws_o);
  float* ws_m = reinterpret_cast<float*>(A.ws_m);
  float* ws_l = reinterpret_cast<float*>(A.ws_l);
  const int lane = threadIdx.x & 31;
  const int hq = A.hq;

  if (first) {
    tk::warp::neg_infty(st.m);
    tk::warp::zero(st.l);
    tk::warp::zero(st.o);
  } else {
    // running state of this query block from the previous chunk
    for_each(st.o, [&](float& v, int rr, int cc) {
      const int row = wrow0 + rr;
      v = (qrow_wg + row < n_rows) ? ws_o[((row_base + row) * hq + h) * kD + cc] : 0.f;
    });
    float* mv = reinterpret_cast<float*>(&sm.vec[2 * wg]);
    float* lv = reinterpret_cast<float*>(&sm.vec[2 * wg + 1]);
    for (int i = threadIdx.x & 127; i < kRows; i += 128) {
      const bool ok = qrow_wg + i < n_rows;
      mv[i] = ok ? ws_m[(row_base + i) * hq + h] : kNegInf;
      lv[i] = ok ? ws_l[(row_base + i) * hq + h] : 0.f;
    }
    tk::warpgroup::sync(2 + wg);
    tk::warpgroup::load(st.m, sm.vec[2 * wg]);
    tk::warpgroup::load(st.l, sm.vec[2 * wg + 1]);
    (void)lane;
  }

  wait_phase(&sm.bar[0], 0, 2, 0);
  const int wg_last_pos = qpos_wg + 63;
  for (int j = p0, it = 0; j < p1; ++j, ++it) {
    const int key0 = j * kRows;
    const uint32_t ph = (it / kStages) & 1;
    if (key0 <= wg_last_pos) {
      const bool masked = key0 + kRows - 1 > qpos_wg || key0 + kRows > kv_len;
      consume_page(st, sm.q[wg], sm.k[it % kStages], sm.v[it % kStages],
                   &sm.bar[1 + it % kStages], &sm.bar[4 + it % kStages], ph, A.scale_log2,
                   masked, wrow0, qpos_wg, key0, kv_len, it);
    } else {
      // pages beyond this warpgroup's causal limit: wait so the ring stays
      // in step with the producer, then release
      wait_phase(&sm.bar[1 + it % kStages], ph, 5, it);
      wait_phase(&sm.bar[4 + it % kStages], ph, 6, it);
    }
    release_page(sm, it);
  }

  if (last) {
    tk::warp::div_row(st.o, st.o, st.l);
    __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(A.out);
    for_each(st.o, [&](float& v, int rr, int cc) {
      const int row = wrow0 + rr;
      if (qrow_wg + row < n_rows)
        out[((row_base + row) * hq + h) * kD + cc] = __float2bfloat16(v);
    });
  } else {
    for_each(st.o, [&](float& v, int rr, int cc) {
      const int row = wrow0 + rr;
      if (qrow_wg + row < n_rows) ws_o[((row_base + row) * hq + h) * kD + cc] = v;
    });
    tk::warpgroup::store(sm.vec[2 * wg], st.m);
    tk::warpgroup::store(sm.vec[2 * wg + 1], st.l);
    tk::warpgroup::sync(2 + wg);
    const float* mv = reinterpret_cast<const float*>(&sm.vec[2 * wg]);
    const float* lv = reinterpret_cast<const float*>(&sm.vec[2 * wg + 1]);
    for (int i = threadIdx.x & 127; i < kRows; i += 128) {
      if (qrow_wg + i < n_rows) {
        ws_m[(row_base + i) * hq + h] = mv[i];
        ws_l[(row_base + i) * hq + h] = lv[i];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------
__device__ __forceinline__ void decode(const TileDesc& t, unsigned char* arena) {
  const AttnArgs& A = *reinterpret_cast<const AttnArgs*>(t.a[0]);
  const int r = t.i0, g = t.i1, c = t.i3;
  const int* info = reinterpret_cast<const int*>(A.req_info) + 4 * r;
  const int q_row = info[0], kv_len = info[3];
  const int chunk_keys = A.chunk_blocks * kRows;
  const int k0 = c * chunk_keys;
  const int k1 = min(kv_len, k0 + chunk_keys);
  const int p0 = k0 / kRows, p1 = (k1 + kRows - 1) / kRows;
  const int n_pages = p1 - p0;
  const int n_chunks = (kv_len + chunk_keys - 1) / chunk_keys;
  const int hq = A.hq, group = A.group;
  Smem sm = carve(arena);
  init_barriers(sm);

  if (threadIdx.x == 0) {
    // both warpgroups use the same query rows: the group's heads
    cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&sm.bar[0], sizeof(q_st));
    cute::SM90_TMA_LOAD_5D::copy(reinterpret_cast<const void*>(A.q_map), &sm.bar[0], 0, &sm.q[0],
                                 0, q_row * hq + g * group, 0, 0, 0);
    produce_pages(A, sm, r, g, p0, p1);
    return;
  }
  if (threadIdx.x < 128) return;

  const int wg = (threadIdx.x - 128) / 128;
  const int wrow0 = 16 * (tk::warpid() % 4);
  Softmax st;
  tk::warp::neg_infty(st.m);
  tk::warp::zero(st.l);
  tk::warp::zero(st.o);
  wait_phase(&sm.bar[0], 0, 2, 0);
  // warpgroup wg takes pages p0 + wg, p0 + wg + 2, ...
  for (int it = 0; it < n_pages; ++it) {
    const uint32_t ph = (it / kStages) & 1;
    const int s = it % kStages;
    if ((it & 1) == wg) {
      const int key0 = (p0 + it) * kRows;
      // rows are heads of one token: only keys at or past kv_len are hidden
      consume_page(st, sm.q[0], sm.k[s], sm.v[s], &sm.bar[1 + s], &sm.bar[4 + s], ph,
                   A.scale_log2, key0 + kRows > kv_len, wrow0, 1 << 29, key0, kv_len, it);
    } else {
      wait_phase(&sm.bar[1 + s], ph, 5, it);
      wait_phase(&sm.bar[4 + s], ph, 6, it);
    }
    release_page(sm, it);
  }

  // merge the two warpgroups' states: stage wg 1 in shared memory
  if (wg == 1) {
    tk::warpgroup::store(sm.o[1], st.o);
    tk::warpgroup::store(sm.vec[0], st.m);
    tk::warpgroup::store(sm.vec[1], st.l);
  }
  cutlass::arch::NamedBarrier::sync(256, 1);
  if (wg == 1) return;
  const float* m1 = reinterpret_cast<const float*>(&sm.vec[0]);
  const float* l1 = reinterpret_cast<const float*>(&sm.vec[1]);
  tk::warpgroup::store(sm.vec[2], st.m);
  tk::warpgroup::store(sm.vec[3], st.l);
  tk::warpgroup::sync(2);
  const float* m0 = reinterpret_cast<const float*>(&sm.vec[2]);
  const float* l0 = reinterpret_cast<const float*>(&sm.vec[3]);
  __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(A.out);
  float* ws_o = reinterpret_cast<float*>(A.ws_o);
  float* ws_m = reinterpret_cast<float*>(A.ws_m);
  const float sl = A.scale_log2;
  for_each(st.o, [&](float& v, int rr, int cc) {
    const int row = wrow0 + rr;   // head within the group
    if (row >= group) return;
    const float ma = m0[row], mb = m1[row];
    const float mx = fmaxf(ma, mb);
    const float wa = exp2f((ma - mx) * sl), wb = exp2f((mb - mx) * sl);
    const float o1 = sm.o[1][{row, cc}];
    const float l = l0[row] * wa + l1[row] * wb;
    const float o = (v * wa + o1 * wb) / l;
    const int head = g * group + row;
    if (n_chunks == 1) {
      out[(static_cast<size_t>(q_row) * hq + head) * kD + cc] = __float2bfloat16(o);
    } else {
      const size_t slot = (static_cast<size_t>(q_row) * hq + head) * A.max_chunks + c;
      ws_o[slot * kD + cc] = o;
      if (cc == 0) ws_m[slot] = mx * sl * 0.69314718056f + logf(l);   // natural log-sum-exp
    }
  });
}

// Merge the chunk partials of every head of decode request i0.
__device__ __forceinline__ void combine(const TileDesc& t) {
  const AttnArgs& A = *reinterpret_cast<const AttnArgs*>(t.a[0]);
  const int r = t.i0, n = t.i1;
  const int q_row = reinterpret_cast<const int*>(A.req_info)[4 * r];
  const int hq = A.hq;
  const float* ws_o = reinterpret_cast<const float*>(A.ws_o);
  const float* ws_m = reinterpret_cast<const float*>(A.ws_m);
  __nv_bfloat16* out = reinterpret_cast<__nv_bfloat16*>(A.out);
  for (int e = threadIdx.x; e < hq * kD; e += blockDim.x) {
    const int head = e / kD, d = e % kD;
    const size_t base = (static_cast<size_t>(q_row) * hq + head) * A.max_chunks;
    float mx = kNegInf;
    for (int c = 0; c < n; ++c) mx = fmaxf(mx, ws_m[base + c]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < n; ++c) {
      const float w = __expf(ws_m[base + c] - mx);
      num += w * ws_o[(base + c) * kD + d];
      den += w;
    }
    out[(static_cast<size_t>(q_row) * hq + head) * kD + d] = __float2bfloat16(num / den);
  }
}

}  // namespace attn
}  // namespace monofab
