"""Load a Hugging Face MoE checkpoint into the fabric's weight layout.

vLLM's model loader fetches the checkpoint (given a Hugging Face id) and
streams its tensors once. Dense tensors go to HBM; every layer's experts
are stacked, as they stream by, into one pinned CPU DRAM tensor per
projection (w13 = [gate; up] as [E, 2I, H], and w2 as [E, H, I]), which
GEMM tiles read over the link. The experts named by the hot tier are copied
to HBM as well. Supported: Qwen3-MoE (bf16) and GLM-4.5 MoE (bf16, or FP8
in the compressed-tensors format with one scale per output channel).
"""
import glob
import os
import re

import torch

from monoserve.runtime.config import MoEConfig
from monoserve.runtime.weights import ModelWeights

_EXPERT = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\."
                     r"(weight|weight_scale)$")


def resolve(model_path, revision=None, cache_dir=None):
    """The checkpoint's local directory; vLLM's loader downloads a Hugging
    Face id."""
    if os.path.isdir(model_path):
        return model_path
    from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf
    return download_weights_from_hf(model_path, cache_dir, ["*.safetensors", "*.json"],
                                    revision=revision)


def safetensors_files(folder):
    from vllm.model_executor.model_loader.weight_utils import (
        filter_duplicate_safetensors_files, filter_files_not_needed_for_inference)
    files = sorted(glob.glob(os.path.join(folder, "*.safetensors")))
    files = filter_duplicate_safetensors_files(files, folder, "model.safetensors.index.json")
    return filter_files_not_needed_for_inference(files)


class _Checkpoint:
    """One pass over the checkpoint through vLLM's weight iterator: routed
    expert tensors go to expert(layer, expert, projection, kind, tensor) as
    they arrive, and the other tensors are kept by name."""

    def __init__(self, folder, expert):
        from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
        self.parts = {}
        for name, t in safetensors_weights_iterator(safetensors_files(folder), False):
            m = _EXPERT.match(name)
            if m:
                expert(int(m[1]), int(m[2]), m[3], m[4], t)
            else:
                self.parts[name] = t

    def has(self, name):
        return name in self.parts

    def get(self, name):
        return self.parts[name]


class _Experts:
    """Pinned per-layer stacks the routed experts stream into, with their
    FP8 scales when the checkpoint has them."""

    def __init__(self, cfg, layers, dtype, scaled, log):
        E, I, H = cfg.num_experts, cfg.intermediate, cfg.hidden
        self.I, self.log = I, log
        self.w13 = {l: torch.empty(E, 2 * I, H, dtype=dtype, pin_memory=True) for l in layers}
        self.w2 = {l: torch.empty(E, H, I, dtype=dtype, pin_memory=True) for l in layers}
        self.s13 = {l: torch.empty(E, 2 * I) for l in layers} if scaled else None
        self.s2 = {l: torch.empty(E, H) for l in layers} if scaled else None
        self.expected = len(self.w13) * E * 3 * (2 if scaled else 1)
        self.count = 0

    def __call__(self, l, e, proj, kind, t):
        if l not in self.w13:   # dense layers have none; a next-token prediction layer is skipped
            return
        I = self.I
        if kind == "weight":
            dst = {"gate": self.w13[l][e, :I], "up": self.w13[l][e, I:], "down": self.w2[l][e]}[proj]
            dst.copy_(t)
        elif self.s13 is not None:
            dst = {"gate": self.s13[l][e, :I], "up": self.s13[l][e, I:], "down": self.s2[l][e]}[proj]
            dst.copy_(t.float().reshape(-1))
        else:
            return
        self.count += 1
        if self.log and self.count % max(1, self.expected // 8) == 0:
            self.log(f"loaded {100 * self.count // self.expected}% of the experts")

    def check(self):
        if self.count != self.expected:
            raise ValueError(f"the checkpoint holds {self.count} of {self.expected} expert tensors")


def load_qwen3_moe(fab, model_path, hot, staging_slots=0, staging_buffers=2, log=print,
                   revision=None, cache_dir=None):
    """hot[l]: experts of layer l to keep in HBM (see monoserve.placement)."""
    from transformers import AutoConfig
    folder = resolve(model_path, revision, cache_dir)
    cfg = MoEConfig.from_hf(AutoConfig.from_pretrained(folder))
    L = cfg.num_layers
    ex = _Experts(cfg, range(L), torch.bfloat16, False, log)
    rd = _Checkpoint(folder, ex)
    ex.check()
    pre = "model.layers.{}."
    embed = rd.get("model.embed_tokens.weight")
    dense = {"embed": embed,
             "lm_head": rd.get("lm_head.weight") if rd.has("lm_head.weight") else embed,
             "final_norm": rd.get("model.norm.weight")}
    for k in ("in_norm", "qkv", "qkv_bias", "q_norm", "k_norm", "o", "post_norm", "router"):
        dense[k] = []
    for l in range(L):
        p = pre.format(l)
        dense["in_norm"].append(rd.get(p + "input_layernorm.weight"))
        dense["qkv"].append(torch.cat([rd.get(p + f"self_attn.{n}_proj.weight") for n in "qkv"]))
        dense["qkv_bias"].append(None)
        dense["q_norm"].append(rd.get(p + "self_attn.q_norm.weight"))
        dense["k_norm"].append(rd.get(p + "self_attn.k_norm.weight"))
        dense["o"].append(rd.get(p + "self_attn.o_proj.weight"))
        dense["post_norm"].append(rd.get(p + "post_attention_layernorm.weight"))
        dense["router"].append(rd.get(p + "mlp.gate.weight"))
    experts = [(ex.w13[l], ex.w2[l]) for l in range(L)]
    return cfg, ModelWeights(fab, cfg, dense, experts, hot, staging_slots=staging_slots,
                             staging_buffers=staging_buffers)


def load_glm4_moe(fab, model_path, hot, staging_slots=0, staging_buffers=2, log=print,
                  revision=None, cache_dir=None):
    """GLM-4.5 MoE: dense leading layers, one shared expert per MoE layer,
    sigmoid routing with a selection bias, QKV bias. The next-token
    prediction layer after the last decoder layer is not loaded."""
    from transformers import AutoConfig
    folder = resolve(model_path, revision, cache_dir)
    cfg = MoEConfig.from_hf(AutoConfig.from_pretrained(folder))
    L = cfg.num_layers
    ex = _Experts(cfg, range(cfg.dense_layers, L),
                  torch.float8_e4m3fn if cfg.fp8 else torch.bfloat16, cfg.fp8, log)
    rd = _Checkpoint(folder, ex)
    ex.check()

    def lin(name):
        """A linear layer's weight and, for FP8, its fp32 scale per row."""
        w = rd.get(name + ".weight")
        s = rd.get(name + ".weight_scale").float().reshape(-1) if rd.has(name + ".weight_scale") else None
        return w, s

    def cat(parts):
        ws, ss = zip(*parts)
        return torch.cat(ws), (torch.cat(ss) if ss[0] is not None else None)

    names = ("in_norm", "qkv", "qkv_scale", "qkv_bias", "q_norm", "k_norm", "o", "o_scale",
             "post_norm", "router", "router_bias", "shared_w13", "shared_w13_scale", "shared_w2",
             "shared_w2_scale", "mlp_w13", "mlp_w13_scale", "mlp_w2", "mlp_w2_scale")
    dense = {"embed": rd.get("model.embed_tokens.weight"), "lm_head": rd.get("lm_head.weight"),
             "final_norm": rd.get("model.norm.weight"), **{n: [None] * L for n in names}}
    experts, scales = [None] * L, [None] * L
    for l in range(L):
        p = f"model.layers.{l}."
        dense["in_norm"][l] = rd.get(p + "input_layernorm.weight")
        dense["post_norm"][l] = rd.get(p + "post_attention_layernorm.weight")
        dense["qkv"][l], dense["qkv_scale"][l] = cat([lin(p + f"self_attn.{n}_proj") for n in "qkv"])
        if cfg.qkv_bias:
            dense["qkv_bias"][l] = torch.cat([rd.get(p + f"self_attn.{n}_proj.bias") for n in "qkv"])
        dense["o"][l], dense["o_scale"][l] = lin(p + "self_attn.o_proj")
        if l < cfg.dense_layers:
            dense["mlp_w13"][l], dense["mlp_w13_scale"][l] = cat(
                [lin(p + "mlp.gate_proj"), lin(p + "mlp.up_proj")])
            dense["mlp_w2"][l], dense["mlp_w2_scale"][l] = lin(p + "mlp.down_proj")
            continue
        dense["router"][l] = rd.get(p + "mlp.gate.weight")
        dense["router_bias"][l] = rd.get(p + "mlp.gate.e_score_correction_bias").float()
        dense["shared_w13"][l], dense["shared_w13_scale"][l] = cat(
            [lin(p + "mlp.shared_experts.gate_proj"), lin(p + "mlp.shared_experts.up_proj")])
        dense["shared_w2"][l], dense["shared_w2_scale"][l] = lin(p + "mlp.shared_experts.down_proj")
        experts[l] = (ex.w13[l], ex.w2[l])
        scales[l] = (ex.s13[l], ex.s2[l]) if ex.s13 is not None else None
    if all(s is None for s in scales):
        scales = None
    return cfg, ModelWeights(fab, cfg, dense, experts, hot, staging_slots=staging_slots,
                             staging_buffers=staging_buffers, expert_scales=scales)


def load_checkpoint(fab, model_path, hot, **kw):
    """(MoEConfig, ModelWeights) of a checkpoint of a supported model type."""
    from transformers import AutoConfig
    folder = resolve(model_path, kw.get("revision"), kw.get("cache_dir"))
    model_type = AutoConfig.from_pretrained(folder).model_type
    if model_type == "qwen3_moe":
        return load_qwen3_moe(fab, folder, hot, **kw)
    if model_type == "glm4_moe":
        return load_glm4_moe(fab, folder, hot, **kw)
    raise NotImplementedError(f"model type {model_type}")
