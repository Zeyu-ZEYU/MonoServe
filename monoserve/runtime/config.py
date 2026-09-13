"""Model shape of a MoE decoder, from a Hugging Face config."""
from dataclasses import dataclass


@dataclass
class MoEConfig:
    hidden: int
    intermediate: int          # per expert
    num_experts: int
    top_k: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    vocab: int
    rope_theta: float = 1e6
    rotary_dim: int = 128
    rms_eps: float = 1e-6
    head_dim: int = 128
    qk_norm: bool = True
    qkv_bias: bool = False
    scoring: str = "softmax"   # or "sigmoid" with a selection bias
    renormalize: bool = True
    routed_scale: float = 1.0
    dense_layers: int = 0          # leading layers with a dense MLP instead of MoE
    dense_intermediate: int = 0    # their MLP width
    shared_intermediate: int = 0   # width of the shared expert of every MoE layer (0: none)
    fp8: bool = False              # linear weights in FP8 with per-channel scales

    @property
    def qkv_dim(self):
        return (self.num_heads + 2 * self.num_kv_heads) * self.head_dim

    @classmethod
    def from_hf(cls, c):
        theta = getattr(c, "rope_theta", None)
        if theta is None:
            theta = (getattr(c, "rope_parameters", None) or {}).get("rope_theta", 1e6)
        if c.model_type == "qwen3_moe":
            return cls(hidden=c.hidden_size, intermediate=c.moe_intermediate_size,
                       num_experts=c.num_experts, top_k=c.num_experts_per_tok,
                       num_layers=c.num_hidden_layers, num_heads=c.num_attention_heads,
                       num_kv_heads=c.num_key_value_heads, vocab=c.vocab_size,
                       rope_theta=float(theta), rotary_dim=c.head_dim, rms_eps=c.rms_norm_eps,
                       head_dim=c.head_dim, qk_norm=True,
                       renormalize=bool(c.norm_topk_prob))
        if c.model_type == "glm4_moe":
            # GLM-4.5: one shared expert per MoE layer, dense leading layers,
            # sigmoid routing with a selection bias, QKV bias, and rotary
            # embedding on part of each head
            head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
            q = getattr(c, "quantization_config", None) or {}
            return cls(hidden=c.hidden_size, intermediate=c.moe_intermediate_size,
                       num_experts=c.n_routed_experts, top_k=c.num_experts_per_tok,
                       num_layers=c.num_hidden_layers, num_heads=c.num_attention_heads,
                       num_kv_heads=c.num_key_value_heads, vocab=c.vocab_size,
                       rope_theta=float(theta),
                       rotary_dim=int(head_dim * getattr(c, "partial_rotary_factor", 1.0)),
                       rms_eps=c.rms_norm_eps, head_dim=head_dim,
                       qk_norm=bool(getattr(c, "use_qk_norm", False)),
                       qkv_bias=bool(getattr(c, "attention_bias", False)), scoring="sigmoid",
                       renormalize=bool(c.norm_topk_prob),
                       routed_scale=float(c.routed_scaling_factor),
                       dense_layers=int(c.first_k_dense_replace),
                       dense_intermediate=c.intermediate_size,
                       shared_intermediate=c.moe_intermediate_size * int(c.n_shared_experts),
                       fp8=q.get("quant_method") in ("compressed-tensors", "fp8"))
        raise NotImplementedError(f"model type {c.model_type}")
