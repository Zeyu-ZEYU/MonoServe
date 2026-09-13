"""Attention tiles inside the fabric against torch, over a paged KV cache:
causal prefill with several KV chunks per query block (the online-softmax
state crosses chunks through HBM), and split-KV decode with its combine
tile."""
import math
import time

import pytest
import torch

from monoserve import _C
from monoserve.fabric import Fabric, Program, Tile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
PREFILL, DECODE, COMBINE = 33, 34, 35
D, PAGE = 128, 64


@pytest.fixture
def fab():
    f = Fabric()
    yield f
    f.stop()


def kv_maps(fab, cache):
    blocks, page, hkv, d = cache.shape
    return fab.tensor_map(cache.data_ptr(), [64, page, 2, hkv, blocks],
                          [hkv * d * 2, 128, d * 2, page * hkv * d * 2], [64, 64, 2, 1, 1])


def paged(k_list, hkv, n_blocks):
    """Scatter per-request K (len, hkv, d) into shuffled pages."""
    cache = torch.zeros(n_blocks, PAGE, hkv, D, dtype=torch.bfloat16, device="cuda")
    perm = torch.randperm(n_blocks).tolist()
    tables, used = [], 0
    for k in k_list:
        n = -(-k.shape[0] // PAGE)
        ids = perm[used:used + n]
        used += n
        for j, b in enumerate(ids):
            rows = k[j * PAGE:(j + 1) * PAGE]
            cache[b, :rows.shape[0]] = rows
        tables.append(ids)
    return cache, tables


def run(fab, stages_tiles):
    p = Program()
    prev = None
    first = None
    for tiles in stages_tiles:
        s = p.stage(tiles)
        if prev is None:
            first = s
        else:
            p.after(prev, s)
        prev = s
    h = p.upload(fab, first=first)
    fab.start()
    fab.publish(map=[0] * fab.num_workers, caps=[0], order=[0], programs=[h])
    t0 = time.time()
    while fab.progress()["lanes"][0]["iterations"] < 1:
        assert time.time() - t0 < 60, "timeout"
        time.sleep(1e-3)
    fab.stop()


def reference(q, k, v, pos0, causal=True):
    """q (n, hq, d), k/v (L, hkv, d); query i sits at position pos0 + i."""
    hq, hkv = q.shape[1], k.shape[1]
    kk = k.float().repeat_interleave(hq // hkv, dim=1)
    vv = v.float().repeat_interleave(hq // hkv, dim=1)
    s = torch.einsum("qhd,khd->hqk", q.float(), kk) / math.sqrt(D)
    qp = torch.arange(q.shape[0], device="cuda")[:, None] + pos0
    kp = torch.arange(k.shape[0], device="cuda")[None, :]
    s = s.masked_fill((kp > qp)[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", s.softmax(-1), vv)


@pytest.mark.parametrize("hq,hkv", [(32, 4), (12, 4)])
def test_prefill(fab, hq, hkv):
    torch.manual_seed(0)
    lens = [300, 777]
    qs = [torch.randn(L, hq, D, device="cuda").to(torch.bfloat16) for L in lens]
    ks = [torch.randn(L, hkv, D, device="cuda").to(torch.bfloat16) for L in lens]
    vs = [torch.randn(L, hkv, D, device="cuda").to(torch.bfloat16) for L in lens]
    kc, tables = paged(ks, hkv, 64)
    vc = torch.zeros_like(kc)
    for b_list, v in zip(tables, vs):
        for j, b in enumerate(b_list):
            rows = v[j * PAGE:(j + 1) * PAGE]
            vc[b, :rows.shape[0]] = rows
    q = torch.cat(qs).contiguous()
    T = q.shape[0]
    out = torch.zeros(T, hq, D, dtype=torch.bfloat16, device="cuda")
    ws_o = torch.zeros(T, hq, D, device="cuda")
    ws_m = torch.zeros(T, hq, device="cuda")
    ws_l = torch.zeros(T, hq, device="cuda")
    bt_stride = max(len(t) for t in tables)
    bt = torch.tensor([t + [0] * (bt_stride - len(t)) for t in tables], dtype=torch.int32, device="cuda")
    starts = [0, lens[0]]
    info = torch.tensor([[starts[i], 0, lens[i], lens[i]] for i in range(2)], dtype=torch.int32, device="cuda")
    qmap = fab.tensor_map(q.data_ptr(), [64, T, 2, hq, 1], [hq * D * 2, 128, D * 2, T * hq * D * 2], [64, 64, 2, 1, 1])
    chunk_blocks = 2
    args = torch.frombuffer(bytearray(_C.attn_args(
        q_map=qmap, k_map=kv_maps(fab, kc), v_map=kv_maps(fab, vc), out=out.data_ptr(),
        ws_o=ws_o.data_ptr(), ws_m=ws_m.data_ptr(), ws_l=ws_l.data_ptr(), block_table=bt.data_ptr(),
        req_info=info.data_ptr(), scale_log2=(1 / math.sqrt(D)) * math.log2(math.e),
        bt_stride=bt_stride, hq=hq, hkv=hkv, chunk_blocks=chunk_blocks, max_chunks=1)),
        dtype=torch.uint8).cuda()
    chunk_keys = chunk_blocks * PAGE
    by_chunk = {}
    for r, L in enumerate(lens):
        for qb in range(-(-L // 128)):
            keys_end = min(L, qb * 128 + 128)
            for c in range(-(-keys_end // chunk_keys)):
                by_chunk.setdefault(c, []).extend(
                    Tile(PREFILL, i=(r, h, qb, c), a=(args.data_ptr(),)) for h in range(hq))
    run(fab, [by_chunk[c] for c in sorted(by_chunk)])
    for r, L in enumerate(lens):
        ref = reference(qs[r], ks[r], vs[r], 0)
        got = out[starts[r]:starts[r] + L].float()
        err = (got - ref).abs().max().item()
        assert err < 2e-2, (r, err)


@pytest.mark.parametrize("chunk_blocks", [4, 64])
def test_decode(fab, chunk_blocks):
    torch.manual_seed(1)
    hq, hkv = 32, 4
    lens = [100, 513, 1000]
    ks = [torch.randn(L, hkv, D, device="cuda").to(torch.bfloat16) for L in lens]
    vs = [torch.randn(L, hkv, D, device="cuda").to(torch.bfloat16) for L in lens]
    kc, tables = paged(ks, hkv, 64)
    vc = torch.zeros_like(kc)
    for b_list, v in zip(tables, vs):
        for j, b in enumerate(b_list):
            rows = v[j * PAGE:(j + 1) * PAGE]
            vc[b, :rows.shape[0]] = rows
    B = len(lens)
    q = torch.randn(B, hq, D, device="cuda").to(torch.bfloat16)
    out = torch.zeros(B, hq, D, dtype=torch.bfloat16, device="cuda")
    max_chunks = 16
    ws_o = torch.zeros(B, hq, max_chunks, D, device="cuda")
    ws_m = torch.zeros(B, hq, max_chunks, device="cuda")
    bt_stride = max(len(t) for t in tables)
    bt = torch.tensor([t + [0] * (bt_stride - len(t)) for t in tables], dtype=torch.int32, device="cuda")
    info = torch.tensor([[i, L - 1, 1, L] for i, L in enumerate(lens)], dtype=torch.int32, device="cuda")
    qmap = fab.tensor_map(q.data_ptr(), [64, B * hq, 2, 1, 1], [D * 2, 128, B * hq * D * 2, B * hq * D * 2], [64, 64, 2, 1, 1])
    args = torch.frombuffer(bytearray(_C.attn_args(
        q_map=qmap, k_map=kv_maps(fab, kc), v_map=kv_maps(fab, vc), out=out.data_ptr(),
        ws_o=ws_o.data_ptr(), ws_m=ws_m.data_ptr(), ws_l=0, block_table=bt.data_ptr(),
        req_info=info.data_ptr(), scale_log2=(1 / math.sqrt(D)) * math.log2(math.e),
        bt_stride=bt_stride, hq=hq, hkv=hkv, chunk_blocks=chunk_blocks, max_chunks=max_chunks)),
        dtype=torch.uint8).cuda()
    chunk_keys = chunk_blocks * PAGE
    dec, comb = [], []
    for r, L in enumerate(lens):
        n = -(-L // chunk_keys)
        dec += [Tile(DECODE, i=(r, g, 0, c), a=(args.data_ptr(),)) for g in range(hkv) for c in range(n)]
        if n > 1:
            comb.append(Tile(COMBINE, i=(r, n), a=(args.data_ptr(),)))
    run(fab, [dec, comb] if comb else [dec])
    for r, L in enumerate(lens):
        ref = reference(q[r:r + 1], ks[r], vs[r], L - 1)
        err = (out[r:r + 1].float() - ref).abs().max().item()
        assert err < 2e-2, (r, err)
