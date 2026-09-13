// Operand blocks of model tiles, shared by the host program builders and
// the device tiles.
#pragma once
#include <cstdint>

namespace monofab {

enum ModelTileKind : uint16_t {
  kTileGemm = 32,          // weight x activation GEMM tile (dense or expert)
  kTileAttnPrefill = 33,   // causal attention, one KV chunk of a query block
  kTileAttnDecode = 34,    // one KV chunk of one request and KV head
  kTileAttnCombine = 35,   // merge the KV-chunk partials of a decode request
  kTileRmsNorm = 36,       // rows: out = rmsnorm(x) * w
  kTileQkRope = 37,        // rows: q/k norm, RoPE, q to its buffer, k/v to the cache
  kTileRouter = 38,        // rows: routing scores, top-k, per-expert counts
  kTileExpand = 39,        // one per lane and layer: expert tiles and stages
  kTilePermute = 40,       // rows: expert-sorted activation rows and pair lists
  kTileAddNorm = 41,       // rows: x = x2 + moe, next norm, zero the accumulator
  kTileEmbed = 42,         // rows: token embedding, first norm, decode bookkeeping
  kTileSample = 43,        // one request row: next token
  kTileAttnPlan = 44,      // one per decode layer: KV-chunk tiles for current lengths
  kTileGemmAsym = 45,      // K-split GEMM tile of a DRAM expert (the asymmetric kernel)
  kTileAsymReduce = 46,    // rows: silu(gate) * up from the asymmetric w13 partial sums
};

// K tiles (of 64) one asymmetric tile keeps in shared memory.
constexpr int kAsymMaxKT = 8;

enum GemmEpilogue : uint32_t {
  kEpiStoreBf16 = 0,     // out[m][n] = acc (+ bias[n]) (+ residual[m][n])
  kEpiSiluMul = 1,       // 64 gate rows and 64 up rows: out[m][f] = silu(g) * u
  kEpiWeightedAdd = 2,   // out32[pair_token[m]][n] += pair_weight[m] * acc
  kEpiStoreF32 = 3,      // out32[m][n] = acc
  kEpiAtomicF32 = 4,     // out32[m][weight row] += acc: K-split partial sums
};

// One GEMM operation (for example a layer's QKV projection or its expert
// w13 GEMMs). A tile of kind kTileGemm computes a 128-row slice of the
// weight's output features for up to bm activation rows:
//   a[0] GemmArgs*, a[1] weight tensor map (K, rows, z), a[2] activation
//   tensor map (K, rows) whose box height is bm,
//   i0 first weight row (feature), i1 first activation row, i2 valid
//   activation rows, i3 weight z coordinate (expert or slot).
// A symmetric tile whose a[3] is nonzero covers only the K range a[3] =
// first K tile | (K tiles << 16) (a K-split GEMM, whose epilogue adds:
// kEpiAtomicF32, the first range adding the bias, or kEpiWeightedAdd).
// An asymmetric tile (kTileGemmAsym) covers one K range of the slice,
// a[3] as above, reads it once, and walks all i2 rows of its expert in
// blocks of bm rows.
struct alignas(64) GemmArgs {
  unsigned long long out;           // bf16 or fp32 output base
  unsigned long long pair_token;    // int32 per activation row (kEpiWeightedAdd)
  unsigned long long pair_weight;   // fp32 per activation row (kEpiWeightedAdd)
  unsigned long long residual;      // bf16, optional (kEpiStoreBf16)
  unsigned long long bias;          // bf16 per feature, optional
  uint32_t ld_out;                  // output row stride, in elements
  uint32_t ld_res;                  // residual row stride, in elements
  uint32_t k_tiles;                 // K / 64
  uint32_t epi;                     // GemmEpilogue
  uint32_t bm;                      // activation rows per tile: 16/32/64/128
  uint32_t n_limit;                 // valid output features
  uint32_t up_offset;               // kEpiSiluMul: row of the first up row
  uint32_t w_fp8;                   // weights are FP8 e4m3; tile a[4]: fp32 scale per weight row
  uint32_t pad[2];
};
static_assert(sizeof(GemmArgs) == 128, "GemmArgs must be 128 bytes");

// Attention over a paged KV cache (head_dim 128, pages of 64 tokens).
//   q_map  prefill: (64, T, 2, Hq, 1) over q[T][Hq][128];
//          decode:  (64, T*Hq, 2, 1, 1) over the same buffer, rows = heads
//   k_map, v_map: (64, 64, 2, Hkv, blocks) over cache[blocks][64][Hkv][128]
//   req_info: int32 [n_req][4] = {first query row, position of that row,
//             query rows, keys visible (positions 0 .. kv_len - 1)}
// Prefill tiles: i0 request, i1 query head, i2 query block of 128 rows,
// i3 KV chunk; chunk c of a block runs after chunk c - 1 and carries the
// online-softmax state (o, m, l) through ws_o / ws_m / ws_l; the last chunk
// writes the normalized output.
// Decode tiles: i0 request, i1 KV head, i3 KV chunk; they write normalized
// partials and their log-sum-exp; a combine tile (i0 request, i1 chunks)
// merges them. With one chunk the decode tile writes the output directly.
struct alignas(64) AttnArgs {
  unsigned long long q_map, k_map, v_map;
  unsigned long long out;          // bf16 [T][Hq][128]
  unsigned long long ws_o;         // fp32 running or partial outputs
  unsigned long long ws_m;         // fp32 running max (prefill) / lse (decode)
  unsigned long long ws_l;         // fp32 running sum (prefill)
  unsigned long long block_table;  // int32 [n_req][bt_stride]
  unsigned long long req_info;     // int32 [n_req][4]
  float scale_log2;                // softmax scale * log2(e)
  uint32_t bt_stride;
  uint32_t hq, hkv, group;
  uint32_t chunk_blocks;           // KV pages per chunk
  uint32_t max_chunks;             // decode partial slots per head
  uint32_t pad[7];
};
static_assert(sizeof(AttnArgs) == 128, "AttnArgs must be 128 bytes");

// Row-wise tiles name rows [i0, i0 + i1).
struct alignas(64) NormArgs {      // kTileRmsNorm, kTileAddNorm, kTileEmbed
  unsigned long long x;            // bf16 [rows][ld] residual stream (written by AddNorm, Embed)
  unsigned long long x2;           // AddNorm: bf16 residual after attention
  unsigned long long acc;          // AddNorm: fp32 MoE output, zeroed after use
  unsigned long long w;            // bf16 [H] norm weight
  unsigned long long out;          // bf16 [rows][ld] normalized rows
  unsigned long long embed;        // Embed: bf16 [vocab][H]
  unsigned long long tokens;       // Embed (prefill): int32 token per row
  unsigned long long gather;       // optional int32 per row: >= 0 copies the row to out2
  unsigned long long out2;         // bf16 [n][H]
  unsigned long long step;         // Embed (decode): StepArgs*
  uint32_t H, ld;
  float eps;
  uint32_t pad[5];
};
static_assert(sizeof(NormArgs) == 128, "NormArgs must be 128 bytes");

// Decode bookkeeping, done by the embedding tiles of every step.
struct alignas(64) StepArgs {
  unsigned long long row_req;      // int32 per row: request slot
  unsigned long long last_token;   // int32 per slot
  unsigned long long seq_len;      // int32 per slot, advanced by one per step
  unsigned long long block_table;  // int32 [slots][bt_stride]
  unsigned long long tok_pos;      // int32 per row (out)
  unsigned long long tok_slot;     // int32 per row (out): page * 64 + offset
  unsigned long long req_info;     // int32 [rows][4] (out), see AttnArgs
  uint32_t bt_stride;
  uint32_t pad[1];
};
static_assert(sizeof(StepArgs) == 64, "StepArgs must be 64 bytes");

struct alignas(64) QkArgs {        // kTileQkRope
  unsigned long long qkv;          // bf16 [T][ld_qkv]: q heads, k heads, v heads
  unsigned long long q_out;        // bf16 [T][hq][128]
  unsigned long long k_cache;      // bf16 [pages][64][hkv][128]
  unsigned long long v_cache;
  unsigned long long q_norm;       // bf16 [128] or 0
  unsigned long long k_norm;
  unsigned long long tok_pos;      // int32 per row
  unsigned long long tok_slot;     // int32 per row
  unsigned long long cos_sin;      // fp32 [max_pos][rot_dim]: cos of the rot_dim/2 angles, then sin; or 0
  uint32_t ld_qkv, hq, hkv, rot_dim;
  float theta, eps;
  uint32_t max_pos;
  uint32_t pad[1];
};
static_assert(sizeof(QkArgs) % 64 == 0, "QkArgs must be a whole number of cache lines");

struct alignas(64) RouterArgs {    // kTileRouter
  unsigned long long logits;       // fp32 [T][E]
  unsigned long long topk_ids;     // int32 [T][K]
  unsigned long long topk_w;       // fp32 [T][K]
  unsigned long long counts;       // int32 [E], accumulated with atomics
  unsigned long long bias;         // fp32 [E] selection bias (sigmoid scoring) or 0
  uint32_t E, K;
  uint32_t scoring;                // 0 softmax, 1 sigmoid
  uint32_t renorm;                 // renormalize the selected weights
  float scale;                     // routed scaling factor
  uint32_t pad[5];
};
static_assert(sizeof(RouterArgs) % 64 == 0, "RouterArgs must be a whole number of cache lines");

// Expansion of one MoE layer of one lane (kTileExpand). The router counted
// the rows routed to every expert; the expander turns the counts into
// expert start offsets, resolves each active expert through the layer's
// indirection table, and writes the expert's GEMM tiles and stage records
// into the program: the w13 stage and w2 stage of expert e are stages
// w13_stage0 + e and w2_stage0 + e, flagged as link stages when the table
// places the expert in CPU DRAM.
struct alignas(64) ExpandArgs {
  unsigned long long counts;       // int32 [E] (zeroed here for the next layer)
  unsigned long long starts;       // int32 [E + 1] (out)
  unsigned long long fill;         // int32 [E] (zeroed here, used by permute)
  unsigned long long table;        // int32 [E][2]: region, slot
  unsigned long long w13_maps;     // u64 [regions]: tensor maps of this layer
  unsigned long long w2_maps;
  unsigned long long xp_maps;      // u64 [4]: activation maps, box 16/32/64/128
  unsigned long long h_maps;
  unsigned long long w13_args;     // u64 [4]: GemmArgs* by tile height
  unsigned long long w2_args;
  unsigned long long hist;         // int32 [E] activation counts of this layer, or 0
  unsigned long long w13_asym_args;  // u64 [4]: GemmArgs* of asymmetric w13 tiles by row block
  unsigned long long w2_asym_args;
  unsigned long long red_args;     // ReduceArgs* of the asymmetric reduce
  unsigned long long w13_scale;    // FP8 weights: fp32 [E][2I] and [E][H] scales, or 0
  unsigned long long w2_scale;
  uint32_t E, H, I;
  uint32_t host_region;            // region id of CPU DRAM
  uint32_t w13_stage0, w2_stage0;  // expert e: stages w13_stage0 + e, red_stage0 + e, w2_stage0 + e
  uint32_t red_stage0;
  uint32_t dyn_first, dyn_cap;     // dynamic tile region of the program
  uint32_t lane;
  uint32_t join_stage;             // waits for every expert's w2 stage; unrouted experts are skipped
  uint32_t pad[5];
};
static_assert(sizeof(ExpandArgs) % 64 == 0, "ExpandArgs must be a whole number of cache lines");

// The reduce of the asymmetric kernel's w13 (kTileAsymReduce, rows
// [i0, i0 + i1)): out = silu(gate) * up from the fp32 partial sums, whose
// rows are then zeroed for the next layer.
struct alignas(64) ReduceArgs {
  unsigned long long ws;           // fp32 [rows][ld_ws]: gate sums, then up sums at column I
  unsigned long long out;          // bf16 [rows][ld_out]
  uint32_t I, ld_ws, ld_out;
  uint32_t pad[9];
};
static_assert(sizeof(ReduceArgs) == 64, "ReduceArgs must be 64 bytes");

struct alignas(64) PermuteArgs {   // kTilePermute
  unsigned long long topk_ids, topk_w, starts, fill;
  unsigned long long pair_token;   // int32 [P] (out)
  unsigned long long pair_weight;  // fp32 [P] (out)
  unsigned long long src;          // bf16 [T][H]
  unsigned long long dst;          // bf16 [P][H]
  uint32_t K, H;
  uint32_t pad[14];
};
static_assert(sizeof(PermuteArgs) == 128, "PermuteArgs must be 128 bytes");

struct alignas(64) SampleArgs {    // kTileSample: i0 row, i1 request slot
  unsigned long long logits;       // fp32 [rows][V]
  unsigned long long temperature;  // fp32 per slot
  unsigned long long last_token;   // int32 per slot (out)
  unsigned long long steps;        // int32 per slot: tokens produced so far
  unsigned long long host_tokens;  // pinned host int32 [slots][ring]: token of step s at s % ring
  unsigned long long host_steps;   // pinned host u64 per slot: tokens produced
  unsigned long long seed;
  uint32_t V;
  uint32_t ring;
};
static_assert(sizeof(SampleArgs) == 64, "SampleArgs must be 64 bytes");

// Decode attention planning (kTileAttnPlan): each step, the embedding tiles
// advance every request's length; this tile then writes the layer's decode
// tiles (one per request, KV head, and chunk of chunk_blocks pages) and the
// combine tiles of requests with more than one chunk into the program's
// dynamic region, as stages stage_attn and stage_comb.
struct alignas(64) AttnPlanArgs {
  unsigned long long attn;         // AttnArgs* of this layer
  unsigned long long req_info;     // int32 [rows][4], written by the embedding tiles
  uint32_t rows;                   // requests in the batch
  uint32_t hkv;
  uint32_t chunk_keys;             // chunk_blocks * 64
  uint32_t stage_attn, stage_comb;
  uint32_t dyn_first, dyn_cap;
  uint32_t pad[7];
};
static_assert(sizeof(AttnPlanArgs) % 64 == 0, "AttnPlanArgs must be a whole number of cache lines");

}  // namespace monofab
