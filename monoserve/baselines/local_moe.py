# SPDX-License-Identifier: Apache-2.0
"""The Local baseline: stock vLLM with the expert set in CPU DRAM and the
same hot tier as MonoServe.

vLLM keeps its scheduler (running requests first, prefill chunks folded
into decode iterations), its model runner with CUDA graphs, and its Triton
fused-MoE kernel; only the experts' location changes. vLLM's UVA offloading
(--cpu-offload-gb with --cpu-offload-params experts) keeps every layer's
expert weights in pinned CPU DRAM, which kernels read over the link. After
the weights are loaded, this plugin copies the hot tier (per layer, the
experts with the highest decode activation probability) into HBM and gives
every MoE layer a table of per-expert weight addresses, through which the
fused-MoE kernel finds each expert's weights.

The kernel is vLLM's fused_moe_kernel (vllm/model_executor/layers/fused_moe/
fused_moe.py, Apache-2.0, Copyright contributors to the vLLM project),
adapted: the expert weight base pointer is loaded from the table instead of
computed from the expert stride, and the paths this baseline does not use
(tensor descriptors, bias, int8/int4 weights, block-wise scales) are
removed.

The plugin is registered as a vLLM general plugin and acts only when vLLM's
additional config has a "local_moe" entry; monoserve.baselines.serve_local
launches vLLM with it.
"""
import logging
import re

import torch

from vllm.triton_utils import tl, triton

log = logging.getLogger(__name__)
_patched = False


@triton.jit
def fused_moe_table_kernel(
    a_ptr, b_table_ptr, c_ptr, a_scale_ptr, b_scale_ptr, topk_weights_ptr, sorted_token_ids_ptr,
    expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, stride_asm, stride_bse,
    stride_bsn,
    naive_block_assignment: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr,
    compute_type: tl.constexpr, use_fp8_w8a8: tl.constexpr, per_channel_quant: tl.constexpr,
    SWAP_AB: tl.constexpr, B_DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    if not naive_block_assignment:
        offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M + offs)
    else:
        offs_token = tl.where(offs == 0, pid_m, num_valid_tokens)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    if off_experts == -1:
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask)
        return

    # the expert's weights, in HBM (hot tier) or in CPU DRAM
    b_base = tl.load(b_table_ptr + off_experts).to(tl.pointer_type(B_DTYPE))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if SWAP_AB:
        a_ptrs = a_ptr + (offs_k[:, None] * stride_ak + offs_token[None, :] // top_k * stride_am)
        b_ptrs = b_base + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)
    else:
        a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_base + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    if use_fp8_w8a8:
        if per_channel_quant:
            b_scale = tl.load(b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn)
            a_scale = tl.load(a_scale_ptr + (offs_token // top_k) * stride_asm, mask=token_mask,
                              other=0.0)[:, None]
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if SWAP_AB:
            a = tl.load(a_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & token_mask[None, :],
                        other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(b, a, acc=accumulator)
        else:
            a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                        other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if SWAP_AB:
        accumulator = tl.trans(accumulator, (1, 0))
    if use_fp8_w8a8:
        accumulator = accumulator * a_scale * b_scale
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]
    tl.store(c_ptrs, accumulator.to(compute_type), mask=c_mask)


def _invoke(A, table, B, C, A_scale, B_scale, topk_weights, sorted_ids, expert_ids, ntp,
            mul_routed_weight, top_k, config, compute_type, fp8):
    from vllm.model_executor.layers.fused_moe.utils import enable_swap_ab
    swap_ab = fp8 and enable_swap_ab(config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"])
    M = A.size(0)
    N, K = B.size(1), B.size(2)
    if sorted_ids is not None:
        EM = sorted_ids.size(0)
        if M < config["BLOCK_SIZE_M"]:
            EM = min(EM, M * top_k * config["BLOCK_SIZE_M"])
    else:
        EM = M * top_k * config["BLOCK_SIZE_M"]
    cfg = {k: v for k, v in config.items() if k != "SPLIT_K"}
    grid = lambda meta: (triton.cdiv(EM, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"]),)  # noqa: E731
    fused_moe_table_kernel[grid](
        A, table, C, A_scale, B_scale, topk_weights, sorted_ids, expert_ids, ntp,
        N, K, EM, M * top_k,
        A.stride(0), A.stride(1), B.stride(2), B.stride(1), C.stride(1), C.stride(2),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        naive_block_assignment=sorted_ids is None, MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k, compute_type=compute_type, use_fp8_w8a8=fp8,
        per_channel_quant=bool(fp8 and B_scale is not None and B_scale.numel() > B.size(0)),
        SWAP_AB=swap_ab, B_DTYPE=tl.float8e4nv if fp8 else tl.bfloat16, **cfg)


def table_fused_experts(layer, x, topk_weights, topk_ids):
    """vLLM's fused-experts computation, with every expert's weights found
    through the layer's address tables."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation, apply_moe_activation
    from vllm.model_executor.layers.fused_moe.config import _get_config_dtype_str
    from vllm.model_executor.layers.fused_moe.fused_moe import (_prepare_expert_assignment,
                                                                try_get_optimal_moe_config)
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
    w1, w2 = layer.w13_weight, layer.w2_weight
    fp8 = w1.dtype == torch.float8_e4m3fn
    M, top_k = x.size(0), topk_ids.size(1)
    E, N, _ = w1.shape
    K = w2.size(1)
    config = try_get_optimal_moe_config(
        w1.size(), w2.size(), top_k,
        _get_config_dtype_str(dtype=x.dtype, use_fp8_w8a8=fp8, use_int8_w8a16=False,
                              use_int4_w4a16=False), M)
    compute_type = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16
    cache1 = torch.empty(M, top_k, N, device=x.device, dtype=x.dtype)
    cache2 = torch.empty(M * top_k, N // 2, device=x.device, dtype=x.dtype)
    cache3 = torch.empty(M, top_k, K, device=x.device, dtype=x.dtype)
    quant = torch.float8_e4m3fn if fp8 else None
    qx, a1_scale = moe_kernel_quantize_input(A=x, A_scale=None, quant_dtype=quant,
                                             per_act_token_quant=fp8, block_shape=None)
    sorted_ids, expert_ids, ntp = _prepare_expert_assignment(topk_ids, config, M, top_k, E, None,
                                                             ignore_invalid_experts=True)
    s13 = layer.w13_weight_scale if fp8 else None
    s2 = layer.w2_weight_scale if fp8 else None
    _invoke(qx, layer._monoserve_table13, w1, cache1, a1_scale, s13, topk_weights, sorted_ids,
            expert_ids, ntp, False, top_k, config, compute_type, fp8)
    act = layer.activation if isinstance(layer.activation, MoEActivation) else \
        MoEActivation.from_str(layer.activation)
    apply_moe_activation(act, cache2, cache1.view(-1, N))
    q2, a2_scale = moe_kernel_quantize_input(A=cache2, A_scale=None, quant_dtype=quant,
                                             per_act_token_quant=fp8, block_shape=None)
    _invoke(q2, layer._monoserve_table2, w2, cache3, a2_scale, s2, topk_weights, sorted_ids,
            expert_ids, ntp, True, 1, config, compute_type, fp8)
    out = torch.empty_like(x)
    ops.moe_sum(cache3, out)
    return out


# ---------------------------------------------------------------------------
# placement
# ---------------------------------------------------------------------------
def _local_config():
    from vllm.config import get_current_vllm_config
    extra = get_current_vllm_config().additional_config or {}
    return extra.get("local_moe") if isinstance(extra, dict) else None


_hot = {}


def _hot_tier(conf, num_layers, num_experts, first_moe_layer):
    """Per layer, the experts to copy into HBM (MonoServe's placement)."""
    key = (num_layers, num_experts)
    if key not in _hot:
        import json

        import numpy as np

        from monoserve.placement import ActivationProfile, hot_tier
        prof = ActivationProfile(num_layers, num_experts)
        if conf.get("profile"):
            with open(conf["profile"]) as f:
                prof.p = np.asarray(json.load(f)["decode"], dtype=np.float64)
        prof.p[:first_moe_layer] = -1.0   # dense layers have no experts
        moe_experts = (num_layers - first_moe_layer) * num_experts
        count = int(conf.get("hot_experts", round(conf.get("hot_fraction", 0.0) * moe_experts)))
        _hot[key] = hot_tier(prof, count)
    return _hot[key]


def place_experts(layer, conf):
    """Expert weights in pinned CPU DRAM (read through unified addressing),
    the layer's hot tier copied into HBM, and the address tables."""
    from vllm.config import get_current_vllm_config
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    hf = get_current_vllm_config().model_config.hf_text_config
    num_layers = hf.num_hidden_layers
    first_moe = getattr(hf, "first_k_dense_replace", 0)
    layer_idx = int(re.search(r"layers\.(\d+)\.", layer.layer_name).group(1))
    E = layer.w13_weight.shape[0]
    hot = _hot_tier(conf, num_layers, E, first_moe)[layer_idx]
    for name, tag in (("w13_weight", "13"), ("w2_weight", "2")):
        p = getattr(layer, name)
        if not getattr(p, "_vllm_is_uva_offloaded", False) and p.data.device.type == "cuda":
            # weight processing left this weight in HBM: back to CPU DRAM
            host = p.data.cpu().pin_memory()
            p.data = get_accelerator_view_from_cpu_tensor(host)
            p._vllm_is_uva_offloaded = True
            setattr(layer, f"_monoserve_host{tag}", host)
            torch.cuda.empty_cache()
        table = [p.data[e].data_ptr() for e in range(E)]
        copy = None
        if hot:
            copy = p.data[torch.tensor(hot, device=p.data.device)].contiguous()
            for slot, e in enumerate(hot):
                table[e] = copy[slot].data_ptr()
        setattr(layer, f"_monoserve_hot{tag}", copy)
        setattr(layer, f"_monoserve_table{tag}",
                torch.tensor(table, dtype=torch.int64, device=p.data.device))


def register():
    """vLLM general plugin: route MoE layers through the address tables when
    the additional config asks for the Local baseline."""
    global _patched
    if _patched:
        return
    _patched = True
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod)
    methods = [UnquantizedFusedMoEMethod]
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 import (  # noqa: E501
            CompressedTensorsW8A8Fp8MoEMethod)
        methods.append(CompressedTensorsW8A8Fp8MoEMethod)
    except ImportError:
        pass
    for cls in methods:
        original = cls.process_weights_after_loading

        def process(self, layer, _original=original):
            _original(self, layer)
            conf = _local_config()
            if conf is not None:
                place_experts(layer, conf)
        cls.process_weights_after_loading = process

    forward = RoutedExperts.forward_modular

    def forward_modular(self, x, topk_weights, topk_ids, shared_experts=None,
                        shared_experts_input=None):
        if getattr(self, "_monoserve_table13", None) is None:
            return forward(self, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
        out = table_fused_experts(self, x, topk_weights, topk_ids)
        if shared_experts is not None:
            from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder
            shared_experts(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
        return out
    RoutedExperts.forward_modular = forward_modular
    log.info("MonoServe Local baseline plugin registered")
