"""A small model with the structure of GLM-4.5 MoE end to end on the fabric:
FP8 weights with per-row scales, a dense first layer, a shared expert beside
the routed ones, sigmoid routing with a selection bias, QKV bias, and rotary
embedding on half of each head. Prefill then decode, in both kernel forms of
the DRAM experts, checked against a torch reference with the same
dequantized weights."""
import time

import pytest
import torch

from monoserve.fabric import Fabric
from monoserve.runtime.config import MoEConfig
from monoserve.runtime.kv import KVCache
from monoserve.runtime.lane import Lane
from monoserve.runtime.requests import RequestTable
from monoserve.runtime.weights import ModelWeights

from test_fabric_model import bf, rms, rope, wait

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

CFG = MoEConfig(hidden=512, intermediate=256, num_experts=16, top_k=4, num_layers=3, num_heads=12,
                num_kv_heads=4, vocab=1024, rope_theta=1e6, rotary_dim=64, qk_norm=False,
                qkv_bias=True, scoring="sigmoid", renormalize=True, routed_scale=1.5,
                dense_layers=1, dense_intermediate=512, shared_intermediate=256, fp8=True)


def ffn(x, w13, w2):
    width = w13.shape[0] // 2
    return bf(torch.nn.functional.silu(x @ w13[:width].t()) * (x @ w13[width:].t())) @ w2.t()


@torch.no_grad()
def reference_logits(w, cfg, tokens):
    L, H, Kt = cfg.num_layers, cfg.hidden, cfg.top_k
    hq, hkv, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    n = len(tokens)
    pos = torch.arange(n, device="cuda")
    x = w.embed[torch.tensor(tokens, device="cuda")].float()
    for l in range(L):
        xn = rms(x, w.in_norm[l], cfg.rms_eps)
        qkv = bf(xn @ w.dequantized("qkv", l).t() + w.qkv_bias[l].float())
        q = bf(rope(qkv[:, :hq * D].view(n, hq, D), pos, cfg.rope_theta, cfg.rotary_dim))
        k = bf(rope(qkv[:, hq * D:(hq + hkv) * D].view(n, hkv, D), pos, cfg.rope_theta, cfg.rotary_dim))
        v = qkv[:, (hq + hkv) * D:].view(n, hkv, D)
        kk, vv = k.repeat_interleave(hq // hkv, 1), v.repeat_interleave(hq // hkv, 1)
        s = torch.einsum("qhd,khd->hqk", q, kk) / D ** 0.5
        s = s.masked_fill(torch.triu(torch.ones(n, n, device="cuda", dtype=torch.bool), 1)[None],
                          float("-inf"))
        ao = bf(torch.einsum("hqk,khd->qhd", s.softmax(-1), vv)).reshape(n, hq * D)
        x2 = bf(x + ao @ w.dequantized("o", l).t())
        xn2 = rms(x2, w.post_norm[l], cfg.rms_eps)
        if l < cfg.dense_layers:
            moe = ffn(xn2, w.dequantized("mlp_w13", l), w.dequantized("mlp_w2", l))
        else:
            p = torch.sigmoid(xn2 @ w.router[l].float().t())
            _, ti = (p + w.router_bias[l].float()).topk(Kt, -1)
            tw = p.gather(1, ti)
            tw = tw / tw.sum(-1, keepdim=True) * cfg.routed_scale
            w13, w2 = w.expert_weights(l)
            moe = ffn(xn2, w.dequantized("shared_w13", l), w.dequantized("shared_w2", l))
            for e in range(cfg.num_experts):
                rows, slot = (ti == e).nonzero(as_tuple=True)
                if len(rows):
                    moe.index_add_(0, rows, tw[rows, slot][:, None] * ffn(xn2[rows], w13[e], w2[e]))
        x = bf(x2 + moe)
    return rms(x, w.final_norm, cfg.rms_eps) @ w.lm_head.float().t()


@pytest.mark.parametrize("form", [64, 0], ids=["sym64", "asym"])
def test_glm_structure(form):
    torch.manual_seed(0)
    fab = Fabric(pool_bytes=512 << 20)
    hot = [[], list(range(8)), list(range(4, 12))]
    w = ModelWeights.synthetic(fab, CFG, hot)
    kv = KVCache(fab, CFG, 64)
    req = RequestTable(fab, 8, 16)
    pre = Lane(fab, CFG, w, kv, req, 0, "prefill", max_tokens=256, max_reqs=4)
    dec = Lane(fab, CFG, w, kv, req, 1, "decode", max_reqs=8, max_len=1024)
    g = torch.Generator().manual_seed(7)
    prompts = {0: torch.randint(0, CFG.vocab, (45,), generator=g).tolist(),
               1: torch.randint(0, CFG.vocab, (130,), generator=g).tolist()}
    pages = {s: kv.allocate(-(-(len(p) + 16) // 64)) for s, p in prompts.items()}
    steps = 5
    n = fab.num_workers
    lanes = [0] * (n // 2) + [1] * (n - n // 2)
    fab.start()
    try:
        for s in prompts:
            req.set_pages(s, pages[s])
            req.set_temperature(s, 0.0)
            req.reset_steps(s)
        hp = pre.build_prefill([dict(slot=s, tokens=prompts[s], pos0=0, pages=pages[s], last=True)
                                for s in prompts])
        fab.publish(map=lanes, caps=[8, 8], order=[0, 1], programs=[hp, 0], forms=[form, form])
        wait(lambda: all(req.produced(s) >= 1 for s in prompts), "prefill")
        for s in prompts:
            req.set_length(s, len(prompts[s]))
        hd = dec.build_decode(list(prompts), pages, iterations=steps)
        fab.publish(map=lanes, caps=[8, 8], order=[1, 0], programs=[hp, hd], forms=[form, form])
        wait(lambda: all(req.produced(s) >= 1 + steps for s in prompts), "decode")
        gen = {s: req.tokens(s, 0, 1 + steps) for s in prompts}
    finally:
        fab.stop()
    for s, prompt in prompts.items():
        logits = reference_logits(w, CFG, prompt + gen[s][:-1])
        for i, t in enumerate(gen[s]):
            lg = logits[len(prompt) - 1 + i]
            best = int(lg.argmax())
            margin = float(lg[best] - lg[t])
            assert t == best or margin < 0.02 * float(lg.max() - lg.min()), (s, i, t, best, margin)
