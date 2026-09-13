"""Qwen3-MoE measurement harness for the Section 2 sweeps and co-runs.

The model runs through Hugging Face transformers with two substitutions
that keep the timed kernels the ones a serving engine runs:

* attention uses vLLM's FlashAttention-3 varlen kernel on a KV cache that
  is pre-allocated in the kernel's native layout, so no concatenation or
  layout copy lands inside a timed span;
* every MoE block becomes FusedMoeBlock, which keeps the model's router and
  runs the expert FFNs with vLLM's fused MoE kernel (weights in HBM) or
  with the symmetric or asymmetric kernel of moe_kernels.py (weights in
  pinned CPU DRAM).

Sublayer spans are timed with CUDA events (common.ModuleTimer). A GPU spin
enqueued before each measured region lets the CPU finish enqueueing first,
so the spans measure back-to-back GPU time. Optional short spins before
each sublayer (spacers, outside the spans) keep the GPU front end's
staging window the same at every SM point.
"""
import gc

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from vllm.vllm_flash_attn import flash_attn_varlen_func

import moe_kernels
from common import log, sleep_cycles
from pinned import to_pinned_cuda_view

ATTN_IMPL_NAME = "vllm_fa3"
# Set if transformers ever passes a materialized attention mask; the
# FlashAttention kernel handles causality, and a 128K x 128K mask would
# not fit in memory.
MASK_SEEN = [False]


# ---------------------------------------------------------------------------
# KV caches
# ---------------------------------------------------------------------------
class FlashStaticCache(Cache):
    """Single-sequence KV cache stored as (max_len, kv_heads, head_dim) per
    layer and written in place. update() returns views of the live rows,
    which FlashAttention consumes without a copy."""

    def __init__(self, num_layers, max_len, kv_heads, head_dim, dtype,
                 device):
        # Cache.__init__ is skipped on purpose: this class provides the few
        # methods the Qwen3-MoE forward path calls.
        self.num_layers = num_layers
        self.max_len = max_len
        self.kbuf = [torch.empty(max_len, kv_heads, head_dim, dtype=dtype,
                                 device=device) for _ in range(num_layers)]
        self.vbuf = [torch.empty(max_len, kv_heads, head_dim, dtype=dtype,
                                 device=device) for _ in range(num_layers)]
        self.pos = 0   # tokens committed for all layers

    def reset_to(self, n):
        self.pos = n

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        s_new = key_states.shape[2]
        start, end = self.pos, self.pos + s_new
        self.kbuf[layer_idx][start:end].copy_(
            key_states[0].transpose(0, 1), non_blocking=True)
        self.vbuf[layer_idx][start:end].copy_(
            value_states[0].transpose(0, 1), non_blocking=True)
        if layer_idx == self.num_layers - 1:
            self.pos = end
        return self.kbuf[layer_idx][:end], self.vbuf[layer_idx][:end]

    def get_mask_sizes(self, cache_position, layer_idx=0):
        q_len = (cache_position.shape[0] if torch.is_tensor(cache_position)
                 else int(cache_position))
        return self.pos + q_len, 0

    @property
    def is_sliding(self):
        return [False] * self.num_layers

    def get_seq_length(self, layer_idx=0):
        return self.pos

    def get_max_cache_shape(self):
        return self.max_len

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.pos

    def __len__(self):
        return self.num_layers

    @property
    def is_compileable(self):
        return False


class BatchStridedCache(Cache):
    """KV cache for a batch of equal-length sequences. Each layer keeps one
    (batch * stride, kv_heads, head_dim) buffer; sequence i owns rows
    [i * stride, i * stride + length). FlashAttention reads it as a paged
    cache with one page per sequence. Prefill fills one sequence at a time
    (active_seq); decode appends one token to every sequence."""

    def __init__(self, num_layers, batch, stride, kv_heads, head_dim, dtype,
                 device):
        stride = -(-stride // 16) * 16   # page size must be a multiple of 16
        self.num_layers = num_layers
        self.batch = batch
        self.stride = stride
        self.kbuf = [torch.empty(batch * stride, kv_heads, head_dim,
                                 dtype=dtype, device=device)
                     for _ in range(num_layers)]
        self.vbuf = [torch.empty(batch * stride, kv_heads, head_dim,
                                 dtype=dtype, device=device)
                     for _ in range(num_layers)]
        self.kview = [b.view(batch, stride, kv_heads, head_dim)
                      for b in self.kbuf]
        self.vview = [b.view(batch, stride, kv_heads, head_dim)
                      for b in self.vbuf]
        self.block_table = torch.arange(batch, device=device,
                                        dtype=torch.int32).unsqueeze(1)
        self.base_idx = torch.arange(batch, device=device,
                                     dtype=torch.int64) * stride
        self.seqused = torch.zeros(batch, device=device, dtype=torch.int32)
        self.cu_q_dec = torch.arange(0, batch + 1, device=device,
                                     dtype=torch.int32)
        self.pos = 0
        self.active_seq = None

    def reset_to(self, n):
        self.pos = n

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.active_seq is not None:
            j = self.active_seq
            s_new = key_states.shape[2]
            valid = self.pos + s_new
            start = j * self.stride + self.pos
            self.kbuf[layer_idx][start:start + s_new].copy_(
                key_states[0].transpose(0, 1), non_blocking=True)
            self.vbuf[layer_idx][start:start + s_new].copy_(
                value_states[0].transpose(0, 1), non_blocking=True)
            if layer_idx == self.num_layers - 1:
                self.pos += s_new
            b0 = j * self.stride
            return (self.kbuf[layer_idx][b0:b0 + valid],
                    self.vbuf[layer_idx][b0:b0 + valid])
        idx = self.base_idx + self.pos
        self.kbuf[layer_idx].index_copy_(
            0, idx, key_states[:, :, 0, :].to(self.kbuf[layer_idx].dtype))
        self.vbuf[layer_idx].index_copy_(
            0, idx, value_states[:, :, 0, :].to(self.vbuf[layer_idx].dtype))
        if layer_idx == 0:
            self.seqused.fill_(self.pos + 1)
        if layer_idx == self.num_layers - 1:
            self.pos += 1
        return self, self   # the attention function reads the buffers

    def get_mask_sizes(self, cache_position, layer_idx=0):
        q_len = (cache_position.shape[0] if torch.is_tensor(cache_position)
                 else int(cache_position))
        return self.pos + q_len, 0

    @property
    def is_sliding(self):
        return [False] * self.num_layers

    def get_seq_length(self, layer_idx=0):
        return self.pos

    def get_max_cache_shape(self):
        return self.stride

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.pos

    def __len__(self):
        return self.num_layers

    @property
    def is_compileable(self):
        return False


# ---------------------------------------------------------------------------
# FlashAttention-3 attention function, registered into transformers
# ---------------------------------------------------------------------------
# cu_seqlens live in persistent device buffers updated with fill_(): building
# them with torch.tensor(..., device="cuda") would be a pageable host copy
# that synchronizes the host with the stream on every layer.
_CU_BUFS = {}


def _cu_seqlens(n, slot, device):
    buf = _CU_BUFS.get(slot)
    if buf is None:
        buf = torch.zeros(2, dtype=torch.int32, device=device)
        _CU_BUFS[slot] = buf
    buf[1].fill_(n)
    return buf


def fa3_attention(module, query, key, value, attention_mask=None,
                  scaling=None, dropout=0.0, **kwargs):
    if attention_mask is not None:
        MASK_SEEN[0] = True
    if isinstance(key, BatchStridedCache):
        cache = key
        b, hq, sq, d = query.shape
        assert sq == 1, "batched cache path is decode only"
        o = flash_attn_varlen_func(
            query[:, :, 0, :], cache.kview[module.layer_idx],
            cache.vview[module.layer_idx], max_seqlen_q=1,
            cu_seqlens_q=cache.cu_q_dec, max_seqlen_k=cache.pos + 1,
            seqused_k=cache.seqused, block_table=cache.block_table,
            softmax_scale=scaling, causal=True, fa_version=3)
        if isinstance(o, tuple):
            o = o[0]
        return o.view(b, 1, hq, d), None

    b, hq, sq, d = query.shape
    assert b == 1, "single-sequence path"
    q = query.transpose(1, 2).reshape(sq, hq, d)
    if key.dim() == 3:          # (kv_len, heads, dim) from FlashStaticCache
        k, v = key, value
    else:                       # (1, heads, kv_len, dim) transformers layout
        sk = key.shape[2]
        k = key.transpose(1, 2).reshape(sk, key.shape[1], d).contiguous()
        v = value.transpose(1, 2).reshape(sk, value.shape[1], d).contiguous()
    sk = k.shape[0]
    o = flash_attn_varlen_func(
        q, k, v, max_seqlen_q=sq, cu_seqlens_q=_cu_seqlens(sq, "q", q.device),
        max_seqlen_k=sk, cu_seqlens_k=_cu_seqlens(sk, "k", q.device),
        softmax_scale=scaling, causal=True, fa_version=3)
    if isinstance(o, tuple):
        o = o[0]
    return o.view(1, sq, hq, d), None


def register_attention():
    ALL_ATTENTION_FUNCTIONS.register(ATTN_IMPL_NAME, fa3_attention)
    # Skip causal-mask materialization for this implementation.
    try:
        from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
        ALL_MASK_ATTENTION_FUNCTIONS.register(ATTN_IMPL_NAME,
                                              lambda *a, **k: None)
    except Exception as e:   # older transformers: checked by validation
        log(f"WARN: mask interface registration failed ({e})")


def use_fa3(model):
    model.config._attn_implementation = ATTN_IMPL_NAME
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = ATTN_IMPL_NAME


# ---------------------------------------------------------------------------
# Fused MoE block
# ---------------------------------------------------------------------------
class FusedMoeBlock(torch.nn.Module):
    """Drop-in replacement for transformers' Qwen3MoeSparseMoeBlock. It
    keeps the original router module and the original stacked expert
    weights, gate_up_proj (E, 2N, K) and down_proj (E, K, N), which is the
    layout vLLM's fused MoE kernel and moe_kernels.py expect."""

    def __init__(self, orig_block, sym_block_m="64"):
        super().__init__()
        self.gate = orig_block.gate
        experts = orig_block.experts
        self.num_experts = experts.num_experts
        self.w13 = experts.gate_up_proj.data
        self.w2 = experts.down_proj.data
        self.kernel = "hbm"
        self.sym_block_m = sym_block_m
        self._pinned = None

    def forward(self, hidden_states):
        b, s, h = hidden_states.shape
        hs = hidden_states.view(-1, h)
        _, topk_w, topk_i = self.gate(hs)
        out = moe_kernels.run_experts(self.kernel, hs, self.w13, self.w2,
                                      topk_w, topk_i, self.num_experts,
                                      self.sym_block_m)
        return out.view(b, s, h)


def convert_moe_layers(model, sym_block_m="64"):
    n = 0
    for layer in model.model.layers:
        blk = layer.mlp
        assert type(blk).__name__ == "Qwen3MoeSparseMoeBlock", \
            type(blk).__name__
        layer.mlp = FusedMoeBlock(blk, sym_block_m)
        n += 1
        del blk
    gc.collect()
    torch.cuda.empty_cache()
    log(f"converted {n} MoE blocks "
        f"(GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB)")


def move_experts_to_cpu(model, kernel):
    """Move every block's expert weights to pinned CPU DRAM, read in place
    by `kernel` ("sym" or "asym"), and free the HBM copies."""
    assert kernel in ("sym", "asym"), kernel
    for layer in model.model.layers:
        blk = layer.mlp
        keep13, v13 = to_pinned_cuda_view(blk.w13)
        keep2, v2 = to_pinned_cuda_view(blk.w2)
        blk._pinned = (keep13, keep2)
        blk.w13, blk.w2 = v13, v2
        blk.kernel = kernel
    torch.cuda.empty_cache()
    log(f"expert weights moved to pinned CPU DRAM, kernel={kernel} "
        f"(GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB)")


def attach_spacers(model, moe_us=0.0, attn_us=0.0):
    """Enqueue a short GPU spin before every MoE and attention sublayer.
    Register before ModuleTimer so the spins fall outside the timed spans.
    Without them a span mixes two regimes: after a long previous kernel the
    front end has pre-staged the next launches, after a short one each
    launch also pays its dispatch latency."""
    def spin(cycles):
        def pre(mod, args):
            torch.cuda._sleep(cycles)
        return pre
    for layer in model.model.layers:
        if moe_us > 0:
            layer.mlp.register_forward_pre_hook(
                spin(sleep_cycles(moe_us / 1e6)))
        if attn_us > 0:
            layer.self_attn.register_forward_pre_hook(
                spin(sleep_cycles(attn_us / 1e6)))


def load_stock_model(model_path):
    """The unmodified model (SDPA attention, eager MoE) on the GPU."""
    register_attention()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa").to(
            "cuda")
    model.eval()
    return model


def prepare(model, experts_in="cpu", moe_kernel="sym", sym_block_m="64"):
    """Switch a stock model to the measured configuration."""
    use_fa3(model)
    convert_moe_layers(model, sym_block_m)
    if experts_in == "cpu":
        move_experts_to_cpu(model, moe_kernel)
    return model


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------
def make_prompt(L, vocab_size, seed, batch=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(1000, vocab_size - 1000, (batch, L),
                         generator=g).cuda()


def forward_call(model, input_ids, cache, cache_position, logits_last_only):
    kwargs = dict(input_ids=input_ids, past_key_values=cache, use_cache=True,
                  cache_position=cache_position)
    if logits_last_only:
        return model(**kwargs, logits_to_keep=1)
    return model(**kwargs)


@torch.inference_mode()
def run_prefill_once(model, timer, cache, ids, fwd_events=None, decouple=0):
    L = ids.shape[1]
    cache.reset_to(0)
    timer.reset()
    cache_position = torch.arange(0, L, device="cuda")
    if decouple:
        torch.cuda._sleep(decouple)   # GPU idles while the CPU enqueues
    if fwd_events:
        fwd_events[0].record()
    out = forward_call(model, ids, cache, cache_position,
                       logits_last_only=True)
    if fwd_events:
        fwd_events[1].record()
    return out


@torch.inference_mode()
def run_decode_steps(model, timer, cache, first_tok, L, n_steps,
                     fwd_events=None, decouple=0):
    """The cache holds L prefill tokens; decode n_steps greedily. With
    decouple > 0 a GPU spin precedes every step so the CPU enqueues the
    whole step first (fwd_events include the spins, sublayer spans do not).
    The next token stays on the GPU, so no host sync enters the loop."""
    cache.reset_to(L)
    timer.reset()
    tok = first_tok
    if fwd_events:
        fwd_events[0].record()
    for i in range(n_steps):
        if decouple:
            torch.cuda._sleep(decouple)
        cache_position = torch.arange(L + i, L + i + 1, device="cuda")
        out = forward_call(model, tok, cache, cache_position,
                           logits_last_only=False)
        tok = out.logits[:, -1:, :].argmax(dim=-1)
    if fwd_events:
        fwd_events[1].record()


@torch.inference_mode()
def prefill_all(model, cache, prompts, timer=None):
    """Prefill the sequences of a BatchStridedCache one at a time and return
    the first decode tokens (B, 1)."""
    toks = []
    L = prompts[0].shape[1]
    for j, ids in enumerate(prompts):
        if timer is not None:
            timer.reset()
        cache.active_seq = j
        cache.reset_to(0)
        out = model(input_ids=ids, past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(0, L, device="cuda"),
                    logits_to_keep=1)
        toks.append(out.logits[:, -1:, :].argmax(dim=-1))
    cache.active_seq = None
    cache.reset_to(L)
    torch.cuda.synchronize()
    return torch.cat(toks, dim=0)


@torch.inference_mode()
def decode_batch(model, timer, cache, tok0, L, n_steps, fwd_events=None,
                 decouple=0, replay=None):
    """Batched decode of n_steps. replay (B, >= n_steps) feeds fixed
    per-sequence tokens: greedy decoding of random prompts degenerates
    within a few steps into repeating one token, which collapses routing
    onto a handful of experts; replaying fixed random tokens keeps routing
    diverse and makes the workload identical across SM points and batch
    sizes."""
    cache.reset_to(L)
    timer.reset()
    tok = tok0 if replay is None else replay[:, 0:1]
    if fwd_events:
        fwd_events[0].record()
    for i in range(n_steps):
        if decouple:
            torch.cuda._sleep(decouple)
        out = model(input_ids=tok, past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(L + i, L + i + 1,
                                                device="cuda"))
        if replay is None:
            tok = out.logits[:, -1:, :].argmax(dim=-1)
        elif i + 1 < n_steps:
            tok = replay[:, i + 1:i + 2]
    if fwd_events:
        fwd_events[1].record()
