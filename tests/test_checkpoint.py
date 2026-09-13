"""The checkpoint loader streams a Hugging Face checkpoint through vLLM's
weight iterator: every expert lands in its layer's pinned stack, the dense
tensors in HBM, all bit for bit."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_streamed_checkpoint(tmp_path):
    pytest.importorskip("vllm")
    from safetensors.torch import load_file
    from tiny_model import make_tiny_qwen3_moe

    from monoserve.fabric import Fabric
    from monoserve.runtime.checkpoint import load_checkpoint
    d = make_tiny_qwen3_moe(None, tmp_path / "tiny", layers=2, experts=8)
    t = load_file(str(d / "model.safetensors"))
    fab = Fabric(pool_bytes=64 << 20)
    cfg, w = load_checkpoint(fab, str(d), [[0, 3], [5]], log=None)
    I = cfg.intermediate
    for l in range(cfg.num_layers):
        p = f"model.layers.{l}."
        for e in range(cfg.num_experts):
            q = p + f"mlp.experts.{e}."
            assert torch.equal(w.host_w13[l][e, :I], t[q + "gate_proj.weight"]), (l, e)
            assert torch.equal(w.host_w13[l][e, I:], t[q + "up_proj.weight"]), (l, e)
            assert torch.equal(w.host_w2[l][e], t[q + "down_proj.weight"]), (l, e)
        assert w.host_w13[l].is_pinned() and w.host_w2[l].is_pinned()
        qkv = torch.cat([t[p + f"self_attn.{n}_proj.weight"] for n in "qkv"])
        assert torch.equal(w.qkv[l].cpu(), qkv)
        assert torch.equal(w.router[l].cpu(), t[p + "mlp.gate.weight"])
    assert torch.equal(w.embed.cpu(), t["model.embed_tokens.weight"])
    assert torch.equal(w.lm_head.cpu(), t["lm_head.weight"])
