"""A small MoE transformer end to end on the fabric: a prefill lane runs two
prompts and samples their first tokens; a decode lane then generates five
more tokens each, on disjoint SM floors, with half of every layer's experts
in the HBM hot tier and the other half read from pinned CPU DRAM through
link-gated tiles. Each generated token must be the argmax of a torch
reference run with teacher forcing, up to a small logit margin."""
import time

import numpy as np
import pytest
import torch

from monoserve.fabric import Fabric
from monoserve.runtime.config import MoEConfig
from monoserve.runtime.kv import KVCache
from monoserve.runtime.lane import Lane
from monoserve.runtime.requests import RequestTable
from monoserve.runtime.weights import ModelWeights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

CFG = MoEConfig(hidden=512, intermediate=256, num_experts=16, top_k=4, num_layers=2,
                num_heads=4, num_kv_heads=2, vocab=1024, rope_theta=1e6)


def bf(t):
    return t.to(torch.bfloat16).float()


def rms(x, w, eps):
    inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return bf(bf(x * inv) * w.float())


def rope(x, pos, theta, rot):
    half = rot // 2
    j = torch.arange(half, device=x.device, dtype=torch.float32)
    inv = 1.0 / torch.pow(torch.tensor(theta, dtype=torch.float32, device=x.device), 2 * j / rot)
    ang = pos[:, None].float() * inv[None]
    c, s = ang.cos()[:, None, :], ang.sin()[:, None, :]
    x1, x2 = x[..., :half], x[..., half:rot]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, x[..., rot:]], -1)


@torch.no_grad()
def reference_logits(w, cfg, tokens):
    L, H, E, Kt = cfg.num_layers, cfg.hidden, cfg.num_experts, cfg.top_k
    hq, hkv, D, I = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.intermediate
    n = len(tokens)
    tok = torch.tensor(tokens, device="cuda")
    pos = torch.arange(n, device="cuda")
    x = w.embed[tok].float()
    for l in range(L):
        xn = rms(x, w.in_norm[l], cfg.rms_eps)
        qkv = bf(xn @ w.qkv[l].float().t())
        q = qkv[:, :hq * D].view(n, hq, D)
        k = qkv[:, hq * D:(hq + hkv) * D].view(n, hkv, D)
        v = qkv[:, (hq + hkv) * D:].view(n, hkv, D)
        q = bf(rope(bf(q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + cfg.rms_eps)) * w.q_norm[l].float(), pos, cfg.rope_theta, cfg.rotary_dim))
        k = bf(rope(bf(k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + cfg.rms_eps)) * w.k_norm[l].float(), pos, cfg.rope_theta, cfg.rotary_dim))
        kk, vv = k.repeat_interleave(hq // hkv, 1), v.repeat_interleave(hq // hkv, 1)
        s = torch.einsum("qhd,khd->hqk", q, kk) / D ** 0.5
        s = s.masked_fill(torch.triu(torch.ones(n, n, device="cuda", dtype=torch.bool), 1)[None], float("-inf"))
        ao = bf(torch.einsum("hqk,khd->qhd", s.softmax(-1), vv)).reshape(n, hq * D)
        x2 = bf(x + ao @ w.o[l].float().t())
        xn2 = rms(x2, w.post_norm[l], cfg.rms_eps)
        p = (xn2 @ w.router[l].float().t()).softmax(-1)
        tw, ti = p.topk(Kt, -1)
        tw = tw / tw.sum(-1, keepdim=True)
        moe = torch.zeros(n, H, device="cuda")
        w13, w2 = w.host_w13[l].cuda().float(), w.host_w2[l].cuda().float()
        for e in range(E):
            rows, slot = (ti == e).nonzero(as_tuple=True)
            if len(rows) == 0:
                continue
            g = xn2[rows] @ w13[e, :I].t()
            u = xn2[rows] @ w13[e, I:].t()
            hh = bf(torch.nn.functional.silu(g) * u)
            moe.index_add_(0, rows, tw[rows, slot][:, None] * (hh @ w2[e].t()))
        x = bf(x2 + moe)
    xn = rms(x, w.final_norm, cfg.rms_eps)
    return xn @ w.lm_head.float().t()


def wait(pred, what, timeout=60):
    t0 = time.time()
    while not pred():
        assert time.time() - t0 < timeout, f"timed out waiting for {what}"
        time.sleep(1e-3)


@pytest.mark.parametrize("form", [64, 0], ids=["sym64", "asym"])
def test_prefill_then_decode(form):
    """form: the kernel form of the DRAM experts, a symmetric template of
    tile height 64 or the asymmetric kernel (0)."""
    torch.manual_seed(0)
    fab = Fabric(pool_bytes=512 << 20)
    hot = [list(range(8)), list(range(4, 12))]
    w = ModelWeights.synthetic(fab, CFG, hot)
    kv = KVCache(fab, CFG, 64)
    req = RequestTable(fab, 8, 16)
    pre = Lane(fab, CFG, w, kv, req, 0, "prefill", max_tokens=256, max_reqs=4)
    dec = Lane(fab, CFG, w, kv, req, 1, "decode", max_reqs=8, max_len=1024)
    g = torch.Generator().manual_seed(3)
    prompts = {0: torch.randint(0, CFG.vocab, (37,), generator=g).tolist(),
               1: torch.randint(0, CFG.vocab, (100,), generator=g).tolist()}
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
        stats = fab.lane_stats(0), fab.lane_stats(1)
    finally:
        fab.stop()
    assert max(s["inflight_max"] for s in stats) <= 8
    for s, prompt in prompts.items():
        logits = reference_logits(w, CFG, prompt + gen[s][:-1])
        for i, t in enumerate(gen[s]):
            lg = logits[len(prompt) - 1 + i]
            best = int(lg.argmax())
            margin = float(lg[best] - lg[t])
            assert t == best or margin < 0.02 * float(lg.max() - lg.min()), (s, i, t, best, margin)
