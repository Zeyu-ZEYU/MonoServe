"""MoE expert kernels for expert weights in CPU DRAM (Section 2.2, Fig. 3).

Both kernels compute the expert FFNs of one MoE layer,

    out[t] = sum_k  w[t, k] * down_e(silu(gate_e(x[t])) * up_e(x[t])),

with e the k-th expert chosen for token t. The activations are read from
HBM. The expert weights are read from wherever their tensors live: HBM,
or pinned host memory read over the link (pinned.py).

Symmetric kernel (SymGEMM). Output stationary, the dataflow of cuBLAS and
CUTLASS. Each program owns one BLOCK_M x BLOCK_N output tile of one expert
and loops over K, so every M block of an expert re-reads that expert's
weights: the link carries w * ceil(M_e / BLOCK_M) bytes for an expert with
M_e routed rows.

Asymmetric kernel (AsymGEMM). Weight stationary with a split along K. Each
program loads one BLOCK_N x BLOCK_K weight tile once and streams all of the
expert's token blocks through it, storing fp32 partial sums of its K slice
to an HBM workspace; a reduction then sums the slices. Each weight byte
crosses the link once, at the cost of extra HBM reads and writes and of
the instructions that issue them.

The two wrappers share the routing layout (vLLM's moe_align_block_size),
the SiLU-and-multiply step, and the weighted top-k sum, so a latency
difference between them comes from the dataflow alone.
"""
import torch
import triton
import triton.language as tl

TILE_HEIGHTS = (16, 32, 64, 128, 256)


def pick_block_m(policy, rows):
    """Symmetric tile height: a fixed value, or "auto" for the smallest
    height in TILE_HEIGHTS that covers `rows` (at most 256)."""
    if policy == "auto":
        return next((c for c in TILE_HEIGHTS if c >= rows), TILE_HEIGHTS[-1])
    return int(policy)


def _align(topk_ids, block_m, num_experts):
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size)
    return moe_align_block_size(topk_ids, block_m, num_experts)


# ---------------------------------------------------------------------------
# Symmetric (output-stationary) grouped GEMM
# ---------------------------------------------------------------------------
@triton.jit
def _sym_moe_gemm(
    x_ptr,             # GEMM1: hidden (M, K) | GEMM2: act (P, K), bf16
    w_ptr,             # (E, N, K) bf16, in HBM or pinned host memory
    out_ptr,           # (P, N) fp32, each element written once
    sorted_ids_ptr,    # (EM_len,) padded pair ids, invalid = P_total
    expert_ids_ptr,    # (EM_len / BLOCK_M,) expert of each padded M block
    num_post_pad_ptr,  # () padded rows in use (device scalar)
    P_total, EM_len,
    N, K,
    stride_xm, stride_xk,
    stride_we, stride_wn, stride_wk,
    stride_om, stride_on,
    TOP_K: tl.constexpr,
    ROW_IS_PAIR: tl.constexpr,  # GEMM2 rows are pairs; GEMM1 rows are tokens
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m * BLOCK_M >= tl.load(num_post_pad_ptr):
        return
    e = tl.load(expert_ids_ptr + pid_m)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pair_ids = tl.load(sorted_ids_ptr + offs_m, mask=offs_m < EM_len,
                       other=P_total)
    m_mask = pair_ids < P_total
    if ROW_IS_PAIR:
        x_rows = pair_ids
    else:
        x_rows = pair_ids // TOP_K

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        x_tile = tl.load(
            x_ptr + x_rows[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        # Every M block of this expert loads this weight tile again.
        w_tile = tl.load(
            w_ptr + e * stride_we + offs_n[:, None] * stride_wn
            + offs_k[None, :] * stride_wk,
            mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.dot(x_tile, tl.trans(w_tile), out_dtype=tl.float32)

    tl.store(out_ptr + pair_ids[:, None] * stride_om
             + offs_n[None, :] * stride_on,
             acc, mask=m_mask[:, None] & n_mask[None, :])


def sym_fused_experts(hidden_states, w13, w2, topk_weights, topk_ids,
                      num_experts, block_m=64, block_n=128, block_k=64):
    """Expert FFNs with the symmetric kernel. hidden_states (M, K) bf16;
    w13 (E, 2N, K) and w2 (E, K, N), in HBM or as pinned-host CUDA views.
    Returns (M, K) bf16."""
    M, K = hidden_states.shape
    E, twoN, _ = w13.shape
    N = twoN // 2
    top_k = topk_ids.shape[1]
    P = M * top_k
    dev = hidden_states.device

    sorted_ids, expert_blk_ids, num_post_pad = _align(topk_ids, block_m,
                                                      num_experts)
    EM_len = sorted_ids.shape[0]
    n_mblocks = triton.cdiv(EM_len, block_m)
    num_warps = 8 if block_m >= 128 else 4

    inter1 = torch.empty(P, twoN, dtype=torch.float32, device=dev)
    _sym_moe_gemm[(n_mblocks, triton.cdiv(twoN, block_n))](
        hidden_states, w13, inter1, sorted_ids, expert_blk_ids, num_post_pad,
        P, EM_len, twoN, K,
        hidden_states.stride(0), hidden_states.stride(1),
        w13.stride(0), w13.stride(1), w13.stride(2),
        inter1.stride(0), inter1.stride(1),
        TOP_K=top_k, ROW_IS_PAIR=False,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=4)

    act = (torch.nn.functional.silu(inter1[:, :N]) * inter1[:, N:]).to(
        hidden_states.dtype)

    out_pairs = torch.empty(P, K, dtype=torch.float32, device=dev)
    _sym_moe_gemm[(n_mblocks, triton.cdiv(K, block_n))](
        act, w2, out_pairs, sorted_ids, expert_blk_ids, num_post_pad,
        P, EM_len, K, N,
        act.stride(0), act.stride(1),
        w2.stride(0), w2.stride(1), w2.stride(2),
        out_pairs.stride(0), out_pairs.stride(1),
        TOP_K=top_k, ROW_IS_PAIR=True,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=4)

    out = (out_pairs.view(M, top_k, K)
           * topk_weights.view(M, top_k, 1).to(torch.float32)).sum(dim=1)
    return out.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Asymmetric (weight-stationary, K-split) grouped GEMM
# ---------------------------------------------------------------------------
@triton.jit
def _asym_moe_gemm(
    x_ptr, w_ptr, ws_ptr,  # ws: (K_TILES, P, N) fp32 partial sums
    sorted_ids_ptr, expert_blk_start_ptr,
    P_total, N, K,
    stride_xm, stride_xk,
    stride_we, stride_wn, stride_wk,
    stride_ws_s, stride_ws_m, stride_ws_n,
    TOP_K: tl.constexpr,
    ROW_IS_PAIR: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    e = tl.program_id(0)
    pid_nk = tl.program_id(1)
    n_tiles = tl.cdiv(N, BLOCK_N)
    pid_n = pid_nk % n_tiles
    pid_k = pid_nk // n_tiles

    blk_start = tl.load(expert_blk_start_ptr + e)
    blk_end = tl.load(expert_blk_start_ptr + e + 1)
    if blk_start == blk_end:
        return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K

    # The weight tile crosses the link exactly once.
    w_tile = tl.load(
        w_ptr + e * stride_we + offs_n[:, None] * stride_wn
        + offs_k[None, :] * stride_wk,
        mask=n_mask[:, None] & k_mask[None, :], other=0.0)

    # 64-bit addressing: long inputs make k_tiles * P * N exceed 2^32.
    ws_base = ws_ptr + pid_k.to(tl.int64) * stride_ws_s
    for blk in range(blk_start, blk_end):
        offs_m = blk * BLOCK_M + tl.arange(0, BLOCK_M)
        pair_ids = tl.load(sorted_ids_ptr + offs_m)
        m_mask = pair_ids < P_total
        if ROW_IS_PAIR:
            x_rows = pair_ids
        else:
            x_rows = pair_ids // TOP_K
        x_tile = tl.load(
            x_ptr + x_rows[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        part = tl.dot(x_tile, tl.trans(w_tile), out_dtype=tl.float32)
        tl.store(ws_base + pair_ids[:, None].to(tl.int64) * stride_ws_m
                 + offs_n[None, :] * stride_ws_n,
                 part, mask=m_mask[:, None] & n_mask[None, :])


def _expert_block_starts(expert_ids, num_post_pad, block_m, num_experts,
                         device):
    """First padded M block of every expert, computed without a host sync
    so it can run inside timed regions and CUDA graphs."""
    n = expert_ids.shape[0]
    valid = (torch.arange(n, device=device, dtype=torch.int64)
             < (num_post_pad.to(torch.int64) // block_m))
    counts = torch.zeros(num_experts, dtype=torch.float32, device=device)
    counts.scatter_add_(0, expert_ids.clamp(0, num_experts - 1).to(
        torch.int64), valid.to(torch.float32))
    starts = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    starts[1:] = torch.cumsum(counts, 0).to(torch.int32)
    return starts


def _asym_stage(x, w, sorted_ids, blk_starts, P, N, K, top_k, row_is_pair,
                block_m, block_n, block_k):
    assert K % block_k == 0, f"K={K} must be a multiple of BLOCK_K={block_k}"
    k_tiles = K // block_k
    ws = torch.empty(k_tiles, P, N, dtype=torch.float32, device=x.device)
    grid = (w.shape[0], triton.cdiv(N, block_n) * k_tiles)
    _asym_moe_gemm[grid](
        x, w, ws, sorted_ids, blk_starts, P, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1), w.stride(2),
        ws.stride(0), ws.stride(1), ws.stride(2),
        TOP_K=top_k, ROW_IS_PAIR=row_is_pair,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=4, num_stages=3)
    # The reduction over K slices is the HBM traffic the asymmetric
    # dataflow trades for its single pass over the weights.
    return ws[0] if k_tiles == 1 else ws.sum(dim=0)


def asym_fused_experts(hidden_states, w13, w2, topk_weights, topk_ids,
                       num_experts, block_m=16, bn1=128, bk1=512, bn2=128,
                       bk2=256):
    """Expert FFNs with the asymmetric kernel; same interface as
    sym_fused_experts."""
    M, K = hidden_states.shape
    E, twoN, _ = w13.shape
    N = twoN // 2
    top_k = topk_ids.shape[1]
    P = M * top_k
    dev = hidden_states.device

    sorted_ids, expert_blk_ids, num_post_pad = _align(topk_ids, block_m,
                                                      num_experts)
    blk_starts = _expert_block_starts(expert_blk_ids, num_post_pad, block_m,
                                      num_experts, dev)
    inter1 = _asym_stage(hidden_states, w13, sorted_ids, blk_starts, P, twoN,
                         K, top_k, False, block_m, bn1, bk1)
    act = (torch.nn.functional.silu(inter1[:, :N]) * inter1[:, N:]).to(
        hidden_states.dtype)
    del inter1
    out_pairs = _asym_stage(act, w2, sorted_ids, blk_starts, P, K, N, top_k,
                            True, block_m, bn2, bk2)
    out = (out_pairs.view(M, top_k, K)
           * topk_weights.view(M, top_k, 1).to(torch.float32)).sum(dim=1)
    return out.to(hidden_states.dtype)


def run_experts(kernel, hidden_states, w13, w2, topk_weights, topk_ids,
                num_experts, sym_block_m="64"):
    """Dispatch by kernel name: "sym", "asym", or "hbm" (vLLM's fused MoE
    kernel on weights in HBM)."""
    tw = topk_weights.to(torch.float32)
    ti = topk_ids.to(torch.int32)
    if kernel == "sym":
        bm = pick_block_m(sym_block_m, hidden_states.shape[0])
        return sym_fused_experts(hidden_states, w13, w2, tw, ti, num_experts,
                                 block_m=bm)
    if kernel == "asym":
        return asym_fused_experts(hidden_states, w13, w2, tw, ti,
                                  num_experts)
    if kernel == "hbm":
        from vllm.model_executor.layers.fused_moe import fused_experts
        return fused_experts(hidden_states, w13, w2, tw, ti,
                             global_num_experts=num_experts)
    raise ValueError(f"unknown MoE kernel {kernel!r}")


def _validate():
    """Both kernels, on HBM weights and on pinned-host weights, against
    vLLM's fused MoE kernel."""
    from pinned import to_pinned_cuda_view
    from vllm.model_executor.layers.fused_moe import fused_experts

    E, K, N, TOPK = 128, 2048, 768, 8   # Qwen3-30B-A3B expert shapes
    g = torch.Generator(device="cpu").manual_seed(7)
    w13 = (torch.randn(E, 2 * N, K, generator=g) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(E, K, N, generator=g) * 0.02).to(torch.bfloat16)
    w13_hbm, w2_hbm = w13.cuda(), w2.cuda()
    keep13, w13_cpu = to_pinned_cuda_view(w13)
    keep2, w2_cpu = to_pinned_cuda_view(w2)

    def cos(a, b):
        return torch.nn.functional.cosine_similarity(
            a.flatten().float(), b.flatten().float(), dim=0).item()

    for M in (1, 4, 64, 1024):
        hs = (torch.randn(M, K, generator=g) * 0.5).to(torch.bfloat16).cuda()
        tw = torch.rand(M, TOPK, generator=g).float().cuda()
        tw = tw / tw.sum(-1, keepdim=True)
        ti = torch.stack([torch.randperm(E, generator=g)[:TOPK]
                          for _ in range(M)]).to(torch.int32).cuda()
        ref = fused_experts(hs, w13_hbm, w2_hbm, tw, ti, global_num_experts=E)
        for name, a, b in (("HBM", w13_hbm, w2_hbm),
                           ("CPU DRAM", w13_cpu, w2_cpu)):
            c_sym = cos(sym_fused_experts(hs, a, b, tw, ti, E,
                                          block_m=pick_block_m("auto", M)),
                        ref)
            c_asym = cos(asym_fused_experts(hs, a, b, tw, ti, E), ref)
            print(f"M={M:5d} weights in {name:8s}: symmetric cos={c_sym:.6f}"
                  f"  asymmetric cos={c_asym:.6f}")
            assert c_sym > 0.999 and c_asym > 0.999
    print("MOE KERNEL VALIDATION PASSED")


if __name__ == "__main__":
    _validate()
