// Row-wise and bookkeeping tiles around the GEMM and attention tiles:
// normalization, rotary embedding and KV-cache append, routing, expert
// expansion and permutation, residual combination, token embedding, and
// sampling. Rows are handled one warp at a time, 8 bf16 per 16-byte load,
// and a lane issues a segment of loads before it uses any of them, so a row
// costs a few memory round trips rather than one per load.
#pragma once
#include <cuda_bf16.h>

#include "fabric/model_types.h"
#include "fabric/runtime.cuh"

namespace monofab {
namespace aux {

using bf16 = __nv_bfloat16;

// 16-byte chunks a lane loads at once: 2048 bf16 values a warp.
constexpr int kSeg = 8;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ void unpack8(const uint4& u, float f[8]) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 t = __bfloat1622float2(h[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

__device__ __forceinline__ void store8(bf16* p, const float f[8]) {
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = u;
}

__device__ __forceinline__ void load4(const bf16* p, float f[4]) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
  const float2 a = __bfloat1622float2(h[0]), b = __bfloat1622float2(h[1]);
  f[0] = a.x; f[1] = a.y; f[2] = b.x; f[3] = b.y;
}

__device__ __forceinline__ void store4(bf16* p, const float f[4]) {
  uint2 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
  h[0] = __floats2bfloat162_rn(f[0], f[1]);
  h[1] = __floats2bfloat162_rn(f[2], f[3]);
  *reinterpret_cast<uint2*>(p) = u;
}

__device__ __forceinline__ float round_bf16(float v) { return __bfloat162float(__float2bfloat16(v)); }

// Sum of squares of a bf16 row of n 16-byte chunks, over the warp.
__device__ __forceinline__ float row_sumsq(const bf16* in, int n) {
  const int lane = threadIdx.x & 31;
  const uint4* in4 = reinterpret_cast<const uint4*>(in);
  float ss = 0.f;
  for (int c0 = lane; c0 < n; c0 += 32 * kSeg) {
    uint4 v[kSeg];
#pragma unroll
    for (int i = 0; i < kSeg; ++i)
      if (c0 + 32 * i < n) v[i] = in4[c0 + 32 * i];
#pragma unroll
    for (int i = 0; i < kSeg; ++i) {
      if (c0 + 32 * i < n) {
        float f[8];
        unpack8(v[i], f);
#pragma unroll
        for (int j = 0; j < 8; ++j) ss += f[j] * f[j];
      }
    }
  }
  return warp_sum(ss);
}

// out = round(in * inv) * w over a row of n 16-byte chunks (and out2 too if
// given).
__device__ __forceinline__ void row_scale(const bf16* in, float inv, const bf16* w, bf16* out,
                                          bf16* out2, int n) {
  const int lane = threadIdx.x & 31;
  const uint4* in4 = reinterpret_cast<const uint4*>(in);
  const uint4* w4 = reinterpret_cast<const uint4*>(w);
  for (int c0 = lane; c0 < n; c0 += 32 * kSeg) {
    uint4 v[kSeg], g[kSeg];
#pragma unroll
    for (int i = 0; i < kSeg; ++i) {
      if (c0 + 32 * i < n) {
        v[i] = in4[c0 + 32 * i];
        g[i] = w4[c0 + 32 * i];
      }
    }
#pragma unroll
    for (int i = 0; i < kSeg; ++i) {
      const int c = c0 + 32 * i;
      if (c < n) {
        float f[8], gg[8];
        unpack8(v[i], f);
        unpack8(g[i], gg);
#pragma unroll
        for (int j = 0; j < 8; ++j) f[j] = round_bf16(f[j] * inv) * gg[j];
        store8(out + 8 * c, f);
        if (out2) store8(out2 + 8 * c, f);
      }
    }
  }
}

// Normalize row `in` (bf16, length H) into `out` (and `out2` if given).
__device__ __forceinline__ void norm_row(const bf16* in, const bf16* w, bf16* out, bf16* out2,
                                         int H, float eps) {
  const float inv = rsqrtf(row_sumsq(in, H / 8) / H + eps);
  row_scale(in, inv, w, out, out2, H / 8);
}

__device__ __forceinline__ void rmsnorm(const TileDesc& t) {
  const NormArgs& A = *reinterpret_cast<const NormArgs*>(t.a[0]);
  const int nw = blockDim.x >> 5;
  for (int r = t.i0 + (threadIdx.x >> 5); r < static_cast<int>(t.i0 + t.i1); r += nw) {
    const size_t o = static_cast<size_t>(r) * A.ld;
    norm_row(reinterpret_cast<const bf16*>(A.x) + o, reinterpret_cast<const bf16*>(A.w),
             reinterpret_cast<bf16*>(A.out) + o, nullptr, A.H, A.eps);
  }
}

// x = x2 + acc (rounded to bf16), then out = rmsnorm(x); the fp32 rows of acc
// are zeroed for the next GEMM that adds into them. The MoE output meets the
// residual stream here, and so does the K-split output projection.
__device__ __forceinline__ void add_norm(const TileDesc& t) {
  const NormArgs& A = *reinterpret_cast<const NormArgs*>(t.a[0]);
  constexpr int kSegA = 4;   // chunks in flight: bf16 residual pieces and their fp32 sums
  const int nw = blockDim.x >> 5, lane = threadIdx.x & 31;
  const int H = static_cast<int>(A.H), n = H / 8;
  const float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int r = t.i0 + (threadIdx.x >> 5); r < static_cast<int>(t.i0 + t.i1); r += nw) {
    const size_t o = static_cast<size_t>(r) * A.ld;
    bf16* x = reinterpret_cast<bf16*>(A.x) + o;
    const uint4* x2 = reinterpret_cast<const uint4*>(reinterpret_cast<const bf16*>(A.x2) + o);
    float4* acc = reinterpret_cast<float4*>(reinterpret_cast<float*>(A.acc) + static_cast<size_t>(r) * H);
    float ss = 0.f;
    for (int c0 = lane; c0 < n; c0 += 32 * kSegA) {
      uint4 v[kSegA];
      float4 a0[kSegA], a1[kSegA];
#pragma unroll
      for (int i = 0; i < kSegA; ++i) {
        const int c = c0 + 32 * i;
        if (c < n) {
          v[i] = x2[c];
          a0[i] = acc[2 * c];
          a1[i] = acc[2 * c + 1];
        }
      }
#pragma unroll
      for (int i = 0; i < kSegA; ++i) {
        const int c = c0 + 32 * i;
        if (c < n) {
          float f[8];
          unpack8(v[i], f);
          const float s[8] = {a0[i].x, a0[i].y, a0[i].z, a0[i].w, a1[i].x, a1[i].y, a1[i].z, a1[i].w};
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            f[j] = round_bf16(f[j] + s[j]);
            ss += f[j] * f[j];
          }
          acc[2 * c] = zero;
          acc[2 * c + 1] = zero;
          store8(x + 8 * c, f);
        }
      }
    }
    const float inv = rsqrtf(warp_sum(ss) / H + A.eps);
    bf16* out2 = nullptr;
    if (A.gather) {
      const int gi = reinterpret_cast<const int*>(A.gather)[r];
      if (gi >= 0) out2 = reinterpret_cast<bf16*>(A.out2) + static_cast<size_t>(gi) * H;
    }
    // x as this lane just wrote it
    row_scale(x, inv, reinterpret_cast<const bf16*>(A.w), reinterpret_cast<bf16*>(A.out) + o, out2, n);
  }
}

// Token embedding and the first norm. In decode (A.step set) the tile also
// advances every request's position and computes its KV-cache slot.
__device__ __forceinline__ void embed(const TileDesc& t) {
  const NormArgs& A = *reinterpret_cast<const NormArgs*>(t.a[0]);
  const StepArgs* St = reinterpret_cast<const StepArgs*>(A.step);
  const int nw = blockDim.x >> 5, lane = threadIdx.x & 31;
  const int H = static_cast<int>(A.H), n = H / 8;
  for (int r = t.i0 + (threadIdx.x >> 5); r < static_cast<int>(t.i0 + t.i1); r += nw) {
    int tok;
    if (St) {
      const int slot = reinterpret_cast<const int*>(St->row_req)[r];
      tok = reinterpret_cast<const int*>(St->last_token)[slot];
      int* seq = reinterpret_cast<int*>(St->seq_len);
      // a request never runs past its block-table row: one that the host
      // has already stopped may run a step or two before its program is
      // replaced, and must not write into another request's pages
      const int pos = min(seq[slot], static_cast<int>(St->bt_stride) * 64 - 1);
      __syncwarp();
      if (lane == 0) {
        seq[slot] = pos + 1;
        const int* bt = reinterpret_cast<const int*>(St->block_table) + static_cast<size_t>(slot) * St->bt_stride;
        reinterpret_cast<int*>(St->tok_pos)[r] = pos;
        reinterpret_cast<int*>(St->tok_slot)[r] = bt[pos / 64] * 64 + pos % 64;
        int* info = reinterpret_cast<int*>(St->req_info) + 4 * r;
        info[0] = r;
        info[1] = pos;
        info[2] = 1;
        info[3] = pos + 1;
      }
    } else {
      tok = reinterpret_cast<const int*>(A.tokens)[r];
    }
    const size_t o = static_cast<size_t>(r) * A.ld;
    const uint4* e4 = reinterpret_cast<const uint4*>(reinterpret_cast<const bf16*>(A.embed) +
                                                     static_cast<size_t>(tok) * H);
    bf16* x = reinterpret_cast<bf16*>(A.x) + o;
    uint4* x4 = reinterpret_cast<uint4*>(x);
    float ss = 0.f;
    for (int c0 = lane; c0 < n; c0 += 32 * kSeg) {
      uint4 v[kSeg];
#pragma unroll
      for (int i = 0; i < kSeg; ++i)
        if (c0 + 32 * i < n) v[i] = e4[c0 + 32 * i];
#pragma unroll
      for (int i = 0; i < kSeg; ++i) {
        if (c0 + 32 * i < n) {
          x4[c0 + 32 * i] = v[i];
          float f[8];
          unpack8(v[i], f);
#pragma unroll
          for (int j = 0; j < 8; ++j) ss += f[j] * f[j];
        }
      }
    }
    const float inv = rsqrtf(warp_sum(ss) / H + A.eps);
    row_scale(x, inv, reinterpret_cast<const bf16*>(A.w), reinterpret_cast<bf16*>(A.out) + o, nullptr, n);
  }
}

// q/k per-head RMSNorm (if weights are given), neox-style RoPE on the first
// rot_dim dimensions, q to its buffer, k and v to their cache pages. The
// tile covers items [i0, i0 + i1) of the (row, head) grid, row-major; one
// warp per item, each lane owning 4 of the 128 dimensions. The angles' cos
// and sin come from the host-built table when there is one.
__device__ __forceinline__ void qk_rope(const TileDesc& t) {
  const QkArgs& A = *reinterpret_cast<const QkArgs*>(t.a[0]);
  const int heads = A.hq + 2 * A.hkv;
  const int nw = blockDim.x >> 5, lane = threadIdx.x & 31;
  const int half = A.rot_dim / 2;
  const float* cs = reinterpret_cast<const float*>(A.cos_sin);
  for (int it = t.i0 + (threadIdx.x >> 5); it < static_cast<int>(t.i0 + t.i1); it += nw) {
    const int r = it / heads, hh = it % heads;
    const bf16* src = reinterpret_cast<const bf16*>(A.qkv) + static_cast<size_t>(r) * A.ld_qkv + hh * 128;
    float v[4];
    {
      const uint2 u = *reinterpret_cast<const uint2*>(src + 4 * lane);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
      const float2 a = __bfloat1622float2(h[0]), b = __bfloat1622float2(h[1]);
      v[0] = a.x; v[1] = a.y; v[2] = b.x; v[3] = b.y;
    }
    const bool is_q = hh < static_cast<int>(A.hq);
    const bool is_k = !is_q && hh < static_cast<int>(A.hq + A.hkv);
    if (is_q || is_k) {
      const bf16* nw8 = reinterpret_cast<const bf16*>(is_q ? A.q_norm : A.k_norm);
      if (nw8) {
        float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
        const float inv = rsqrtf(warp_sum(ss) / 128.f + A.eps);
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = round_bf16(v[i] * inv) * __bfloat162float(nw8[4 * lane + i]);
      }
      const int pos = reinterpret_cast<const int*>(A.tok_pos)[r];
      float p[4];
      const int d0 = 4 * lane;
      const int partner = d0 < half ? lane + half / 4 : lane - half / 4;
#pragma unroll
      for (int i = 0; i < 4; ++i) p[i] = __shfl_sync(0xffffffffu, v[i], partner & 31);
      if (d0 < static_cast<int>(A.rot_dim)) {
        const float* row = cs ? cs + static_cast<size_t>(min(pos, static_cast<int>(A.max_pos) - 1)) * A.rot_dim
                              : nullptr;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int d = d0 + i;
          const int j = d < half ? d : d - half;
          float s, c;
          if (row) {
            c = row[j];
            s = row[half + j];
          } else {
            const float inv_freq = 1.f / powf(A.theta, (2.f * j) / A.rot_dim);
            sincosf(static_cast<float>(pos) * inv_freq, &s, &c);
          }
          v[i] = d < half ? v[i] * c - p[i] * s : v[i] * c + p[i] * s;
        }
      }
    }
    bf16* dst;
    if (is_q) {
      dst = reinterpret_cast<bf16*>(A.q_out) + (static_cast<size_t>(r) * A.hq + hh) * 128;
    } else {
      const int slot = reinterpret_cast<const int*>(A.tok_slot)[r];
      const int kv_head = is_k ? hh - A.hq : hh - A.hq - A.hkv;
      bf16* cache = reinterpret_cast<bf16*>(is_k ? A.k_cache : A.v_cache);
      dst = cache + ((static_cast<size_t>(slot) * A.hkv) + kv_head) * 128;
    }
    uint2 u;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
    h[0] = __floats2bfloat162_rn(v[0], v[1]);
    h[1] = __floats2bfloat162_rn(v[2], v[3]);
    *reinterpret_cast<uint2*>(dst + 4 * lane) = u;
  }
}

// Routing: softmax (or sigmoid with a selection bias) over E experts, top-K,
// optional renormalization, and per-expert row counts. The logits are
// zeroed once read, since the router GEMM adds its K-split partial sums
// into them. Lane k of the warp keeps the k-th pick.
__device__ __forceinline__ void router(const TileDesc& t) {
  const RouterArgs& A = *reinterpret_cast<const RouterArgs*>(t.a[0]);
  const int nw = blockDim.x >> 5, lane = threadIdx.x & 31;
  constexpr int kMaxPer = 8;   // up to 256 experts
  constexpr int kMaxK = 16;
  const int E = static_cast<int>(A.E), K = static_cast<int>(A.K);
  const float* bias = reinterpret_cast<const float*>(A.bias);
  for (int r = t.i0 + (threadIdx.x >> 5); r < static_cast<int>(t.i0 + t.i1); r += nw) {
    float* lg = reinterpret_cast<float*>(A.logits) + static_cast<size_t>(r) * E;
    float p[kMaxPer], sel[kMaxPer];
#pragma unroll
    for (int i = 0; i < kMaxPer; ++i) {
      const int e = lane + 32 * i;
      p[i] = e < E ? lg[e] : -1e30f;
    }
#pragma unroll
    for (int i = 0; i < kMaxPer; ++i)
      if (lane + 32 * i < E) lg[lane + 32 * i] = 0.f;
    if (A.scoring == 0) {
      float mx = -1e30f;
#pragma unroll
      for (int i = 0; i < kMaxPer; ++i) mx = fmaxf(mx, p[i]);
#pragma unroll
      for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
      float s = 0.f;
#pragma unroll
      for (int i = 0; i < kMaxPer; ++i) {
        p[i] = __expf(p[i] - mx);
        s += p[i];
      }
      s = warp_sum(s);
#pragma unroll
      for (int i = 0; i < kMaxPer; ++i) {
        p[i] /= s;
        sel[i] = lane + 32 * i < E ? p[i] : -1e30f;
      }
    } else {
#pragma unroll
      for (int i = 0; i < kMaxPer; ++i) {
        const int e = lane + 32 * i;
        p[i] = 1.f / (1.f + __expf(-p[i]));
        sel[i] = e < E ? p[i] + (bias ? bias[e] : 0.f) : -1e30f;
      }
    }
    int my_id = 0;
    float my_w = 0.f, wsum = 0.f;
#pragma unroll
    for (int k = 0; k < kMaxK; ++k) {
      if (k >= K) break;
      float bv = -1e30f;
      int bi = 0x7fffffff;
#pragma unroll
      for (int i = 0; i < kMaxPer; ++i) {
        const int e = lane + 32 * i;
        if (sel[i] > bv || (sel[i] == bv && e < bi)) {
          bv = sel[i];
          bi = e;
        }
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) {
          bv = ov;
          bi = oi;
        }
      }
      // the pick's weight, from the lane that holds it
      float mine = 0.f;
#pragma unroll
      for (int i = 0; i < kMaxPer; ++i) {
        if (i == (bi >> 5)) {
          mine = p[i];
          if (lane == (bi & 31)) sel[i] = -1e30f;
        }
      }
      const float w = __shfl_sync(0xffffffffu, mine, bi & 31);
      if (lane == k) {
        my_id = bi;
        my_w = w;
      }
      wsum += w;
    }
    if (lane < K) {
      reinterpret_cast<int*>(A.topk_ids)[static_cast<size_t>(r) * K + lane] = my_id;
      reinterpret_cast<float*>(A.topk_w)[static_cast<size_t>(r) * K + lane] =
          (A.renorm ? my_w / wsum : my_w) * A.scale;
      atomicAdd(reinterpret_cast<int*>(A.counts) + my_id, 1);
    }
  }
}

__device__ __forceinline__ int tile_class(int bm) { return bm <= 16 ? 0 : bm <= 32 ? 1 : bm <= 64 ? 2 : 3; }

// Exclusive prefix sums of v[0, n) into out[0, n], out[n] the total, by one
// warp (n <= 256); returns how many of the values are nonzero.
__device__ __forceinline__ int warp_scan(const int* v, int* out, int n) {
  const int lane = threadIdx.x & 31;
  int loc[8], sum = 0, nz = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = 8 * lane + i;
    loc[i] = e < n ? v[e] : 0;
    sum += loc[i];
    nz += loc[i] != 0;
  }
  int incl = sum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += y;
  }
  int run = incl - sum;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = 8 * lane + i;
    if (e < n) out[e] = run;
    run += loc[i];
  }
  if (lane == 31) out[n] = incl;
#pragma unroll
  for (int o = 16; o; o >>= 1) nz += __shfl_xor_sync(0xffffffffu, nz, o);
  return nz;
}

// One tile per lane and MoE layer: expert offsets, expert tiles, stages.
__device__ __forceinline__ void expand(const TileDesc& t, const ProgramDesc* P, DeviceState& S) {
  const ExpandArgs& A = *reinterpret_cast<const ExpandArgs*>(t.a[0]);
  __shared__ int s_cnt[256];
  __shared__ int s_start[257];
  __shared__ int s_n[256];
  __shared__ int s_off[257];
  __shared__ int s_form;
  __shared__ int s_active, s_skipped;
  int* counts = reinterpret_cast<int*>(A.counts);
  int* starts = reinterpret_cast<int*>(A.starts);
  const int E = A.E;
  if (E > 256) __trap();
  for (int e = threadIdx.x; e < E; e += blockDim.x) s_cnt[e] = counts[e];
  if (threadIdx.x == 32) {
    s_skipped = 0;
    const unsigned long long ep = ld_acquire(&S.published.v);
    const uint32_t f = ld_volatile(&S.snap[ep & 1].form[A.lane]);
    // 0: the asymmetric kernel; otherwise a symmetric template's height
    s_form = f == 0 || f == 16 || f == 32 || f == 64 || f == 128 ? static_cast<int>(f) : 64;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    const int active = warp_scan(s_cnt, s_start, E);
    if (threadIdx.x == 0) s_active = active;
  }
  __syncthreads();
  for (int e = threadIdx.x; e <= E; e += blockDim.x) starts[e] = s_start[e];
  const int* table = reinterpret_cast<const int*>(A.table);
  // K splits of the asymmetric kernel: even ranges of at most kAsymMaxKT
  const int kt13 = static_cast<int>(A.H) / 64, kt2 = static_cast<int>(A.I) / 64;
  const int sp13 = (kt13 + kAsymMaxKT - 1) / kAsymMaxKT, sp2 = (kt2 + kAsymMaxKT - 1) / kAsymMaxKT;
  const int per13 = (kt13 + sp13 - 1) / sp13, per2 = (kt2 + sp2 - 1) / sp2;
  const int ch13 = (kt13 + per13 - 1) / per13, ch2 = (kt2 + per2 - 1) / per2;
  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    const int M = s_cnt[e];
    const bool link = table[2 * e] == static_cast<int>(A.host_region);
    int n = 0;
    if (M > 0 && link && s_form == 0) {
      n = (static_cast<int>(A.I) / 64) * ch13 + (M + 63) / 64 + (static_cast<int>(A.H) / 128) * ch2;
    } else if (M > 0) {
      const int bm = link && s_form ? s_form : M <= 16 ? 16 : M <= 32 ? 32 : M <= 64 ? 64 : 128;
      n = ((M + bm - 1) / bm) * (A.I / 64 + A.H / 128);
    }
    s_n[e] = n;
  }
  __syncthreads();
  if (threadIdx.x < 32) warp_scan(s_n, s_off, E);
  // Close the expert stages' previous activation before their tiles and
  // counts change: a claim through a stale ring entry then finds tag 0
  // instead of the old tag with the new count.
  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    atomicExch(&P->runtime[A.w13_stage0 + e].tag_next, 0ull);
    atomicExch(&P->runtime[A.red_stage0 + e].tag_next, 0ull);
    atomicExch(&P->runtime[A.w2_stage0 + e].tag_next, 0ull);
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0 && s_off[E] > static_cast<int>(A.dyn_cap)) __trap();
  const unsigned long long* w13_args = reinterpret_cast<const unsigned long long*>(A.w13_args);
  const unsigned long long* w2_args = reinterpret_cast<const unsigned long long*>(A.w2_args);
  const unsigned long long* w13_asym = reinterpret_cast<const unsigned long long*>(A.w13_asym_args);
  const unsigned long long* w2_asym = reinterpret_cast<const unsigned long long*>(A.w2_asym_args);
  const unsigned long long* xp_maps = reinterpret_cast<const unsigned long long*>(A.xp_maps);
  const unsigned long long* h_maps = reinterpret_cast<const unsigned long long*>(A.h_maps);
  const unsigned long long* w13_maps = reinterpret_cast<const unsigned long long*>(A.w13_maps);
  const unsigned long long* w2_maps = reinterpret_cast<const unsigned long long*>(A.w2_maps);
  for (int e = threadIdx.x; e < E; e += blockDim.x) {
    const int M = s_cnt[e];
    const int region = table[2 * e], slot = table[2 * e + 1];
    const bool link = region == static_cast<int>(A.host_region);
    const uint32_t base = A.dyn_first + s_off[e];
    StageStatic& s13 = P->stages[A.w13_stage0 + e];
    StageStatic& sr = P->stages[A.red_stage0 + e];
    StageStatic& s2 = P->stages[A.w2_stage0 + e];
    TileDesc d{};
    uint32_t k = base;
    if (M > 0 && link && s_form == 0) {
      // asymmetric: each tile takes one K range and all M rows of the expert
      const int bm = M <= 16 ? 16 : M <= 32 ? 32 : 64;
      const int cls = tile_class(bm);
      d.kind = kTileGemmAsym;
      d.i1 = s_start[e]; d.i2 = M; d.i3 = slot;
      d.a[0] = w13_asym[cls]; d.a[1] = w13_maps[region]; d.a[2] = xp_maps[cls];
      d.a[4] = A.w13_scale ? A.w13_scale + static_cast<unsigned long long>(e) * 2 * A.I * 4 : 0;
      for (int f = 0; f < static_cast<int>(A.I); f += 64)
        for (int k0 = 0; k0 < kt13; k0 += per13) {
          d.i0 = f;
          d.a[3] = static_cast<unsigned long long>(k0) |
                   (static_cast<unsigned long long>(min(per13, kt13 - k0)) << 16);
          P->tiles[k++] = d;
        }
      const uint32_t n13 = k - base;
      TileDesc r{};
      r.kind = kTileAsymReduce;
      r.a[0] = A.red_args;
      for (int m = 0; m < M; m += 64) {
        r.i0 = s_start[e] + m; r.i1 = min(64, M - m);
        P->tiles[k++] = r;
      }
      const uint32_t nred = k - base - n13;
      d.a[0] = w2_asym[cls]; d.a[1] = w2_maps[region]; d.a[2] = h_maps[cls];
      d.a[4] = A.w2_scale ? A.w2_scale + static_cast<unsigned long long>(e) * A.H * 4 : 0;
      for (int n = 0; n < static_cast<int>(A.H); n += 128)
        for (int k0 = 0; k0 < kt2; k0 += per2) {
          d.i0 = n;
          d.a[3] = static_cast<unsigned long long>(k0) |
                   (static_cast<unsigned long long>(min(per2, kt2 - k0)) << 16);
          P->tiles[k++] = d;
        }
      s13.first = base; s13.count = n13; s13.flags = kStageLink;
      sr.first = base + n13; sr.count = nred; sr.flags = 0u;
      s2.first = base + n13 + nred; s2.count = k - base - n13 - nred; s2.flags = kStageLink;
    } else {
      // symmetric: DRAM experts take the lane's template height, HBM
      // experts the smallest height covering their rows
      const int bm = link && s_form ? s_form : M <= 16 ? 16 : M <= 32 ? 32 : M <= 64 ? 64 : 128;
      const int cls = tile_class(bm);
      const int blocks = M > 0 ? (M + bm - 1) / bm : 0;
      const int n13 = blocks * (A.I / 64), n2 = blocks * (A.H / 128);
      d.kind = kTileGemm;
      for (int b = 0; b < blocks; ++b) {
        const int m0 = s_start[e] + b * bm, mv = min(bm, M - b * bm);
        for (int f = 0; f < static_cast<int>(A.I); f += 64) {
          d.i0 = f; d.i1 = m0; d.i2 = mv; d.i3 = slot;
          d.a[0] = w13_args[cls]; d.a[1] = w13_maps[region]; d.a[2] = xp_maps[cls];
          d.a[4] = A.w13_scale ? A.w13_scale + static_cast<unsigned long long>(e) * 2 * A.I * 4 : 0;
          P->tiles[k++] = d;
        }
      }
      for (int b = 0; b < blocks; ++b) {
        const int m0 = s_start[e] + b * bm, mv = min(bm, M - b * bm);
        for (int n = 0; n < static_cast<int>(A.H); n += 128) {
          d.i0 = n; d.i1 = m0; d.i2 = mv; d.i3 = slot;
          d.a[0] = w2_args[cls]; d.a[1] = w2_maps[region]; d.a[2] = h_maps[cls];
          d.a[4] = A.w2_scale ? A.w2_scale + static_cast<unsigned long long>(e) * A.H * 4 : 0;
          P->tiles[k++] = d;
        }
      }
      s13.first = base;
      s13.count = n13;
      s13.flags = link ? kStageLink : 0u;
      if (M == 0 && s_active > 0) {
        // an unrouted expert: its chain (w13, reduce, w2) is passed over,
        // and the join stage no longer waits for it
        s13.flags = kStageSkip;
        atomicAdd(&s_skipped, 1);
      }
      sr.first = base + n13;
      sr.count = 0;
      sr.flags = 0u;
      s2.first = base + n13;
      s2.count = n2;
      s2.flags = link ? kStageLink : 0u;
    }
    if (A.hist && M > 0) atomicAdd(reinterpret_cast<int*>(A.hist) + e, M);
    counts[e] = 0;
    reinterpret_cast<int*>(A.fill)[e] = 0;
  }
  __threadfence();
  __syncthreads();
  // the stage flags are visible before the join stage's count drops; the
  // routed experts' chains keep the count above zero until they complete
  if (threadIdx.x == 0 && s_skipped > 0) atomicSub(&P->runtime[A.join_stage].deps, s_skipped);
}

// The reduce of the asymmetric w13: h = silu(gate) * up from the K-split
// partial sums of rows [i0, i0 + i1), whose partial sums are then zeroed.
__device__ __forceinline__ void asym_reduce(const TileDesc& t) {
  const ReduceArgs& A = *reinterpret_cast<const ReduceArgs*>(t.a[0]);
  const int nw = blockDim.x >> 5, lane = threadIdx.x & 31;
  const int I = static_cast<int>(A.I);
  const float4 zero = make_float4(0.f, 0.f, 0.f, 0.f);
  for (int r = t.i0 + (threadIdx.x >> 5); r < static_cast<int>(t.i0 + t.i1); r += nw) {
    float* g = reinterpret_cast<float*>(A.ws) + static_cast<size_t>(r) * A.ld_ws;
    float* u = g + I;
    bf16* h = reinterpret_cast<bf16*>(A.out) + static_cast<size_t>(r) * A.ld_out;
    for (int c = lane; c < I / 8; c += 32) {
      float4* g4 = reinterpret_cast<float4*>(g + 8 * c);
      float4* u4 = reinterpret_cast<float4*>(u + 8 * c);
      const float4 ga = g4[0], gb = g4[1], ua = u4[0], ub = u4[1];
      const float gv[8] = {ga.x, ga.y, ga.z, ga.w, gb.x, gb.y, gb.z, gb.w};
      const float uv[8] = {ua.x, ua.y, ua.z, ua.w, ub.x, ub.y, ub.z, ub.w};
      float o[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) o[i] = gv[i] / (1.f + __expf(-gv[i])) * uv[i];
      store8(h + 8 * c, o);
      g4[0] = zero; g4[1] = zero; u4[0] = zero; u4[1] = zero;
    }
  }
}

// Expert-sorted activation rows and the pair lists the w2 epilogue uses.
// Every (row, pick) item of the tile first takes its position, one thread
// each; then a warp reads a row once and writes it to the positions of its
// picks.
__device__ __forceinline__ void permute(const TileDesc& t) {
  const PermuteArgs& A = *reinterpret_cast<const PermuteArgs*>(t.a[0]);
  constexpr int kMaxItems = 16 * 16;   // row tiles of 16 rows, at most 16 picks a row
  __shared__ int s_pos[kMaxItems];
  const int K = static_cast<int>(A.K);
  const int rows = static_cast<int>(t.i1);
  const int items = rows * K;
  if (items > kMaxItems) __trap();
  for (int it = threadIdx.x; it < items; it += blockDim.x) {
    const size_t pk = static_cast<size_t>(t.i0) * K + it;
    const int e = reinterpret_cast<const int*>(A.topk_ids)[pk];
    const int pos = reinterpret_cast<const int*>(A.starts)[e] + atomicAdd(reinterpret_cast<int*>(A.fill) + e, 1);
    reinterpret_cast<int*>(A.pair_token)[pos] = static_cast<int>(t.i0) + it / K;
    reinterpret_cast<float*>(A.pair_weight)[pos] = reinterpret_cast<const float*>(A.topk_w)[pk];
    s_pos[it] = pos;
  }
  __syncthreads();
  const int nw = blockDim.x >> 5, lane = threadIdx.x & 31;
  const int n = static_cast<int>(A.H) / 8;
  for (int r = threadIdx.x >> 5; r < rows; r += nw) {
    const uint4* src = reinterpret_cast<const uint4*>(reinterpret_cast<const bf16*>(A.src) +
                                                      static_cast<size_t>(t.i0 + r) * A.H);
    for (int c0 = lane; c0 < n; c0 += 32 * kSeg) {
      uint4 v[kSeg];
#pragma unroll
      for (int i = 0; i < kSeg; ++i)
        if (c0 + 32 * i < n) v[i] = src[c0 + 32 * i];
      for (int k = 0; k < K; ++k) {
        uint4* dst = reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(A.dst) +
                                              static_cast<size_t>(s_pos[r * K + k]) * A.H);
#pragma unroll
        for (int i = 0; i < kSeg; ++i)
          if (c0 + 32 * i < n) dst[c0 + 32 * i] = v[i];
      }
    }
  }
}

// Decode tiles of one layer for the lengths of this step.
__device__ __forceinline__ void attn_plan(const TileDesc& t, const ProgramDesc* P) {
  const AttnPlanArgs& A = *reinterpret_cast<const AttnPlanArgs*>(t.a[0]);
  __shared__ int s_n[1024];
  __shared__ int s_off[1025];
  __shared__ int s_comb;
  const int* info = reinterpret_cast<const int*>(A.req_info);
  const int rows = min(static_cast<int>(A.rows), 1024);
  if (threadIdx.x == 0) {
    // close the stages' previous activation before their tiles and counts
    // change (see expand)
    atomicExch(&P->runtime[A.stage_attn].tag_next, 0ull);
    atomicExch(&P->runtime[A.stage_comb].tag_next, 0ull);
    __threadfence();
  }
  for (int r = threadIdx.x; r < rows; r += blockDim.x) {
    const int kv_len = info[4 * r + 3];
    s_n[r] = (kv_len + A.chunk_keys - 1) / A.chunk_keys;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int acc = 0, comb = 0;
    for (int r = 0; r < rows; ++r) {
      s_off[r] = acc;
      acc += s_n[r] * A.hkv;
      comb += s_n[r] > 1;
    }
    s_off[rows] = acc;
    s_comb = comb;
    if (acc + comb > static_cast<int>(A.dyn_cap)) __trap();
  }
  __syncthreads();
  TileDesc d{};
  d.a[0] = A.attn;
  for (int r = threadIdx.x; r < rows; r += blockDim.x) {
    uint32_t k = A.dyn_first + s_off[r];
    d.kind = kTileAttnDecode;
    d.i0 = r;
    d.i2 = 0;
    for (int g = 0; g < static_cast<int>(A.hkv); ++g)
      for (int c = 0; c < s_n[r]; ++c) {
        d.i1 = g;
        d.i3 = c;
        P->tiles[k++] = d;
      }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t k = A.dyn_first + s_off[rows];
    d.kind = kTileAttnCombine;
    for (int r = 0; r < rows; ++r) {
      if (s_n[r] > 1) {
        d.i0 = r;
        d.i1 = s_n[r];
        d.i3 = 0;
        P->tiles[k++] = d;
      }
    }
    StageStatic& sa = P->stages[A.stage_attn];
    StageStatic& sc = P->stages[A.stage_comb];
    sa.first = A.dyn_first;
    sa.count = s_off[rows];
    sc.first = A.dyn_first + s_off[rows];
    sc.count = s_comb;
  }
  __threadfence();
}

__device__ __forceinline__ unsigned long long mix64(unsigned long long z) {
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

// Next token of one request: greedy at temperature 0, otherwise Gumbel-max
// sampling from softmax(logits / temperature).
__device__ __forceinline__ void sample(const TileDesc& t) {
  const SampleArgs& A = *reinterpret_cast<const SampleArgs*>(t.a[0]);
  const int row = t.i0, slot = t.i1;
  const float* lg = reinterpret_cast<const float*>(A.logits) + static_cast<size_t>(row) * A.V;
  const float temp = reinterpret_cast<const float*>(A.temperature)[slot];
  int* steps = reinterpret_cast<int*>(A.steps);
  const int step = steps[slot];
  float bv = -INFINITY;
  int bi = 0;
  auto consider = [&](float v, int i) {
    if (temp > 0.f) {
      const unsigned long long h = mix64(A.seed ^ (static_cast<unsigned long long>(slot) << 40) ^
                                         (static_cast<unsigned long long>(step) << 20) ^ i);
      const float u = (static_cast<float>(h >> 40) + 0.5f) * (1.f / 16777216.f);
      v = v / temp - __logf(-__logf(u));
    }
    if (v > bv || (v == bv && i < bi)) {
      bv = v;
      bi = i;
    }
  };
  // four logits per load and several loads in flight per thread
  const int V = static_cast<int>(A.V);
  const int V4 = reinterpret_cast<uintptr_t>(lg) % 16 == 0 ? V / 4 : 0;
  const float4* lg4 = reinterpret_cast<const float4*>(lg);
#pragma unroll 4
  for (int i = threadIdx.x; i < V4; i += blockDim.x) {
    const float4 q = __ldcs(lg4 + i);
    consider(q.x, 4 * i);
    consider(q.y, 4 * i + 1);
    consider(q.z, 4 * i + 2);
    consider(q.w, 4 * i + 3);
  }
  for (int i = 4 * V4 + threadIdx.x; i < V; i += blockDim.x) consider(lg[i], i);
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) {
      bv = ov;
      bi = oi;
    }
  }
  __shared__ float sv[32];
  __shared__ int si[32];
  if ((threadIdx.x & 31) == 0) {
    sv[threadIdx.x >> 5] = bv;
    si[threadIdx.x >> 5] = bi;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int w = 1; w < static_cast<int>(blockDim.x >> 5); ++w)
      if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) {
        bv = sv[w];
        bi = si[w];
      }
    reinterpret_cast<int*>(A.last_token)[slot] = bi;
    steps[slot] = step + 1;
    if (A.host_tokens) {
      reinterpret_cast<volatile int*>(A.host_tokens)[static_cast<size_t>(slot) * A.ring + step % A.ring] = bi;
      st_release_sys(reinterpret_cast<unsigned long long*>(A.host_steps) + slot,
                     static_cast<unsigned long long>(step + 1));
    }
  }
}

}  // namespace aux
}  // namespace monofab
