"""A small random Qwen3-MoE checkpoint in Hugging Face format, for tests
that go through vLLM: a real checkpoint's config with few layers and narrow
dimensions, random weights, and the real tokenizer."""
import json
import shutil
from pathlib import Path

import torch

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
# the fields a Qwen3-MoE config needs when no source checkpoint lends one
BASE_CONFIG = {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
               "vocab_size": 1024, "rope_theta": 1000000.0, "rms_norm_eps": 1e-6,
               "max_position_embeddings": 32768, "norm_topk_prob": True, "hidden_act": "silu",
               "attention_bias": False, "bos_token_id": 0, "eos_token_id": 1}


def make_tiny_qwen3_moe(src, dst, layers=2, hidden=512, intermediate=256, experts=16, top_k=4,
                        heads=4, kv_heads=2, seed=0):
    """src: a Qwen3-MoE checkpoint directory whose config and tokenizer the
    small model borrows, or None for a config of its own and no tokenizer."""
    from safetensors.torch import save_file
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if src is None:
        cfg = dict(BASE_CONFIG)
    else:
        src = Path(src)
        cfg = json.loads((src / "config.json").read_text())
        if cfg.get("model_type") != "qwen3_moe":
            raise ValueError("the source checkpoint must be a Qwen3-MoE model")
    D = 128
    cfg.update(num_hidden_layers=layers, hidden_size=hidden, moe_intermediate_size=intermediate,
               intermediate_size=2 * intermediate, num_experts=experts, num_experts_per_tok=top_k,
               num_attention_heads=heads, num_key_value_heads=kv_heads, head_dim=D,
               max_window_layers=layers, mlp_only_layers=[], decoder_sparse_step=1,
               tie_word_embeddings=False, torch_dtype="bfloat16",
               max_position_embeddings=min(cfg.get("max_position_embeddings", 32768), 32768))
    (dst / "config.json").write_text(json.dumps(cfg, indent=1))
    for name in TOKENIZER_FILES if src is not None else ():
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    g = torch.Generator().manual_seed(seed)
    V, H, I = cfg["vocab_size"], hidden, intermediate

    def r(*shape, scale=0.02):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    def ones(n):
        return (1 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)

    t = {"model.embed_tokens.weight": r(V, H, scale=0.5), "lm_head.weight": r(V, H),
         "model.norm.weight": ones(H)}
    for layer in range(layers):
        p = f"model.layers.{layer}."
        t[p + "input_layernorm.weight"] = ones(H)
        t[p + "post_attention_layernorm.weight"] = ones(H)
        t[p + "self_attn.q_proj.weight"] = r(heads * D, H)
        t[p + "self_attn.k_proj.weight"] = r(kv_heads * D, H)
        t[p + "self_attn.v_proj.weight"] = r(kv_heads * D, H)
        t[p + "self_attn.o_proj.weight"] = r(H, heads * D)
        t[p + "self_attn.q_norm.weight"] = ones(D)
        t[p + "self_attn.k_norm.weight"] = ones(D)
        t[p + "mlp.gate.weight"] = r(experts, H, scale=0.1)
        for e in range(experts):
            q = p + f"mlp.experts.{e}."
            t[q + "gate_proj.weight"] = r(I, H)
            t[q + "up_proj.weight"] = r(I, H)
            t[q + "down_proj.weight"] = r(H, I)
    save_file(t, str(dst / "model.safetensors"))
    return dst
