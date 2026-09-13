"""GEMM tiles inside the fabric against torch, with the weights in HBM and
in pinned CPU DRAM read over the link, for every epilogue and tile height.
"""
import pytest
import torch

from monoserve import _C
from monoserve.fabric import Fabric, Program, Tile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

KIND_GEMM, KIND_GEMM_ASYM, KIND_ASYM_REDUCE = 32, 45, 46
EPI_STORE, EPI_SILU_MUL, EPI_WADD, EPI_F32, EPI_ATOMIC = 0, 1, 2, 3, 4


def pinned_view(t):
    """Pinned host copy of t and a CUDA tensor aliasing it (zero copy)."""
    from torch.utils.cpp_extension import load_inline
    global _pv
    try:
        _pv
    except NameError:
        _pv = load_inline(name="t_pinned_view", cpp_sources=r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
torch::Tensor view(torch::Tensor p) {
  void* d = nullptr;
  TORCH_CHECK(cudaHostGetDevicePointer(&d, p.data_ptr(), 0) == cudaSuccess);
  return torch::from_blob(d, p.sizes(), p.strides(), p.options().device(torch::kCUDA));
}""", functions=["view"], with_cuda=True)
    pin = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    pin.copy_(t)
    return pin, _pv.view(pin)


def weight_map(fab, w):   # w: (E, rows, K) bf16, contiguous
    E, rows, K = w.shape
    return fab.tensor_map(w.data_ptr(), [K, rows, E], [K * 2, rows * K * 2], [64, 64, 1])


def act_map(fab, x, bm):  # x: (M, K) bf16
    M, K = x.shape
    return fab.tensor_map(x.data_ptr(), [K, M], [x.stride(0) * 2], [64, bm])


def args_tensor(**kw):
    return torch.frombuffer(bytearray(_C.gemm_args(**kw)), dtype=torch.uint8).cuda()


@pytest.fixture
def fab():
    f = Fabric()
    yield f
    f.stop()


def run_program(fab, tiles):
    p = Program()
    p.stage(tiles)
    h = p.upload(fab, first=0)
    fab.start()
    fab.publish(map=[0] * fab.num_workers, caps=[0], order=[0], programs=[h])
    import time
    t0 = time.time()
    while fab.progress()["lanes"][0]["iterations"] < 1:
        assert time.time() - t0 < 60, "program did not finish"
        time.sleep(1e-3)
    fab.stop()


@pytest.mark.parametrize("where", ["hbm", "cpu"])
@pytest.mark.parametrize("bm", [16, 32, 64, 128])
def test_dense_store(fab, where, bm):
    torch.manual_seed(0)
    E, N, K, M = 2, 256, 512, 200
    w = (torch.randn(E, N, K, device="cuda") * 0.05).to(torch.bfloat16)
    x = (torch.randn(M, K, device="cuda")).to(torch.bfloat16)
    res = torch.randn(M, N, device="cuda").to(torch.bfloat16)
    out = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    keep = None
    if where == "cpu":
        keep, wv = pinned_view(w)
    else:
        wv = w
    e = 1
    args = args_tensor(out=out.data_ptr(), ld_out=N, k_tiles=K // 64, epi=EPI_STORE, bm=bm,
                       n_limit=N, residual=res.data_ptr(), ld_res=N)
    wm, xm = weight_map(fab, wv), act_map(fab, x, bm)
    tiles = [Tile(KIND_GEMM, i=(n0, m0, min(bm, M - m0), e), a=(args.data_ptr(), wm, xm))
             for n0 in range(0, N, 128) for m0 in range(0, M, bm)]
    run_program(fab, tiles)
    ref = (x.float() @ w[e].float().t() + res.float())
    err = (out.float() - ref).abs().max().item()
    assert err < 0.05 * ref.abs().max().item(), err
    del keep


@pytest.mark.parametrize("where", ["hbm", "cpu"])
def test_expert_ffn(fab, where):
    """w13 tiles with the SiLU-and-multiply epilogue, then w2 tiles that add
    routing-weighted outputs into the tokens' fp32 accumulators."""
    torch.manual_seed(1)
    E, H, I, T, TOPK = 8, 512, 256, 64, 2
    w13 = (torch.randn(E, 2 * I, H, device="cuda") * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device="cuda") * 0.05).to(torch.bfloat16)
    x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
    topk = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(T)])
    tw = torch.rand(T, TOPK, device="cuda")
    # expert-sorted pairs: permuted activation rows and their token / weight
    pairs = [(int(topk[t, k]), t, float(tw[t, k])) for t in range(T) for k in range(TOPK)]
    pairs.sort()
    xp = torch.stack([x[t] for _, t, _ in pairs])
    ptok = torch.tensor([t for _, t, _ in pairs], dtype=torch.int32, device="cuda")
    pw = torch.tensor([wt for _, _, wt in pairs], dtype=torch.float32, device="cuda")
    starts = {}
    for i, (e, _, _) in enumerate(pairs):
        starts.setdefault(e, [i, 0])[1] += 1
    hbuf = torch.zeros(len(pairs), I, device="cuda", dtype=torch.bfloat16)
    out32 = torch.zeros(T, H, device="cuda", dtype=torch.float32)
    keep = []
    if where == "cpu":
        k1, w13v = pinned_view(w13)
        k2, w2v = pinned_view(w2)
        keep = [k1, k2]
    else:
        w13v, w2v = w13, w2
    bm = 16
    a13 = args_tensor(out=hbuf.data_ptr(), ld_out=I, k_tiles=H // 64, epi=EPI_SILU_MUL, bm=bm,
                      n_limit=I, up_offset=I)
    a2 = args_tensor(out=out32.data_ptr(), ld_out=H, k_tiles=I // 64, epi=EPI_WADD, bm=bm,
                     n_limit=H, pair_token=ptok.data_ptr(), pair_weight=pw.data_ptr())
    m13, m2 = weight_map(fab, w13v), weight_map(fab, w2v)
    mx, mh = act_map(fab, xp, bm), act_map(fab, hbuf, bm)
    p = Program()
    t13, t2 = [], []
    for e, (s0, cnt) in starts.items():
        for m0 in range(s0, s0 + cnt, bm):
            mv = min(bm, s0 + cnt - m0)
            t13 += [Tile(KIND_GEMM, i=(f0, m0, mv, e), a=(a13.data_ptr(), m13, mx))
                    for f0 in range(0, I, 64)]
            t2 += [Tile(KIND_GEMM, i=(n0, m0, mv, e), a=(a2.data_ptr(), m2, mh))
                   for n0 in range(0, H, 128)]
    s13 = p.stage(t13)
    s2 = p.stage(t2)
    p.after(s13, s2)
    h = p.upload(fab, first=s13)
    fab.start()
    fab.publish(map=[0] * fab.num_workers, caps=[0], order=[0], programs=[h])
    import time
    t0 = time.time()
    while fab.progress()["lanes"][0]["iterations"] < 1:
        assert time.time() - t0 < 60
        time.sleep(1e-3)
    fab.stop()
    ref = torch.zeros(T, H, device="cuda")
    for t in range(T):
        for k in range(TOPK):
            e = int(topk[t, k])
            g = x[t].float() @ w13[e, :I].float().t()
            u = x[t].float() @ w13[e, I:].float().t()
            hh = (torch.nn.functional.silu(g) * u).to(torch.bfloat16).float()
            ref[t] += float(tw[t, k]) * (hh @ w2[e].float().t())
    err = (out32 - ref).abs().max().item()
    assert err < 0.03 * ref.abs().max().item(), err
    del keep


def k_splits(kt, per=8):
    """Even K ranges of at most `per` K tiles, as the expander cuts them."""
    n = -(-kt // per)
    step = -(-kt // n)
    return [(k0, min(step, kt - k0)) for k0 in range(0, kt, step)]


@pytest.mark.parametrize("where", ["hbm", "cpu"])
@pytest.mark.parametrize("bm", [16, 64])
def test_asymmetric_expert_ffn(fab, where, bm):
    """The asymmetric form: K-split w13 tiles into fp32 partial sums, the
    reduce tile, and K-split w2 tiles into the accumulators. Every tile
    covers all rows of its expert and reads its weight slice once."""
    from monoserve.fabric.args import device_block
    torch.manual_seed(2)
    E, H, I, T, TOPK = 4, 1024, 640, 48, 2   # w13: 16 K tiles, w2: 10, two splits each
    w13 = (torch.randn(E, 2 * I, H, device="cuda") * 0.03).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device="cuda") * 0.03).to(torch.bfloat16)
    x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
    topk = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(T)])
    tw = torch.rand(T, TOPK, device="cuda")
    pairs = sorted((int(topk[t, k]), t, float(tw[t, k])) for t in range(T) for k in range(TOPK))
    xp = torch.stack([x[t] for _, t, _ in pairs])
    ptok = torch.tensor([t for _, t, _ in pairs], dtype=torch.int32, device="cuda")
    pw = torch.tensor([wt for _, _, wt in pairs], dtype=torch.float32, device="cuda")
    starts = {}
    for i, (e, _, _) in enumerate(pairs):
        starts.setdefault(e, [i, 0])[1] += 1
    ws = torch.zeros(len(pairs), 2 * I, device="cuda", dtype=torch.float32)
    hbuf = torch.zeros(len(pairs), I, device="cuda", dtype=torch.bfloat16)
    out32 = torch.zeros(T, H, device="cuda", dtype=torch.float32)
    keep = []
    if where == "cpu":
        k1, w13v = pinned_view(w13)
        k2, w2v = pinned_view(w2)
        keep = [k1, k2]
    else:
        w13v, w2v = w13, w2
    a13 = args_tensor(out=ws.data_ptr(), ld_out=2 * I, k_tiles=H // 64, epi=EPI_ATOMIC, bm=bm,
                      n_limit=I, up_offset=I)
    a2 = args_tensor(out=out32.data_ptr(), ld_out=H, k_tiles=I // 64, epi=EPI_WADD, bm=bm,
                     n_limit=H, pair_token=ptok.data_ptr(), pair_weight=pw.data_ptr())
    red = device_block("ReduceArgs", ws=ws, out=hbuf, I=I, ld_ws=2 * I, ld_out=I)
    m13, m2 = weight_map(fab, w13v), weight_map(fab, w2v)
    mx, mh = act_map(fab, xp, bm), act_map(fab, hbuf, bm)
    t13, tr, t2 = [], [], []
    for e, (s0, cnt) in starts.items():
        t13 += [Tile(KIND_GEMM_ASYM, i=(f0, s0, cnt, e), a=(a13.data_ptr(), m13, mx, k0 | (kt << 16)))
                for f0 in range(0, I, 64) for k0, kt in k_splits(H // 64)]
        tr += [Tile(KIND_ASYM_REDUCE, i=(r0, min(64, s0 + cnt - r0)), a=(red.data_ptr(),))
               for r0 in range(s0, s0 + cnt, 64)]
        t2 += [Tile(KIND_GEMM_ASYM, i=(n0, s0, cnt, e), a=(a2.data_ptr(), m2, mh, k0 | (kt << 16)))
               for n0 in range(0, H, 128) for k0, kt in k_splits(I // 64)]
    p = Program()
    s13, sr, s2 = p.stage(t13), p.stage(tr), p.stage(t2)
    p.after(s13, sr)
    p.after(sr, s2)
    h = p.upload(fab, first=s13)
    fab.start()
    fab.publish(map=[0] * fab.num_workers, caps=[0], order=[0], programs=[h])
    import time
    t0 = time.time()
    while fab.progress()["lanes"][0]["iterations"] < 1:
        assert time.time() - t0 < 60
        time.sleep(1e-3)
    fab.stop()
    e_of = torch.tensor([e for e, _, _ in pairs], device="cuda")
    g = torch.einsum("mh,mih->mi", xp.float(), w13[e_of, :I].float())
    u = torch.einsum("mh,mih->mi", xp.float(), w13[e_of, I:].float())
    h_ref = torch.nn.functional.silu(g) * u
    assert (hbuf.float() - h_ref).abs().max().item() < 0.02 * h_ref.abs().max().item()
    assert ws.abs().max().item() == 0.0   # the reduce zeroed the partial sums
    y = torch.einsum("mi,mhi->mh", h_ref.to(torch.bfloat16).float(), w2[e_of].float()) * pw[:, None]
    ref = torch.zeros(T, H, device="cuda").index_add_(0, ptok.long(), y)
    assert (out32 - ref).abs().max().item() < 0.03 * ref.abs().max().item()
    del keep


def quantize(w):
    """Per-row FP8 e4m3: (raw bytes on the GPU, fp32 scale per row, and the
    dequantized weight for references)."""
    s = (w.float().abs().amax(dim=-1, keepdim=True) / 448.0).clamp_min(1e-12)
    q = (w.float() / s).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).contiguous(), s.squeeze(-1).contiguous(), q.float() * s


def weight_map_fp8(fab, w8):   # w8: (E, rows, K) raw e4m3 bytes
    E, rows, K = w8.shape
    return fab.tensor_map(w8.data_ptr(), [K, rows, E], [K, rows * K], [64, 64, 1], dtype=2, swizzle=0)


@pytest.mark.parametrize("where", ["hbm", "cpu"])
@pytest.mark.parametrize("bm", [16, 64, 128])
def test_dense_store_fp8(fab, where, bm):
    """FP8 weights with per-row scales and a bias, bf16 activations."""
    torch.manual_seed(3)
    E, N, K, M = 2, 256, 512, 200
    w8, sc, wd = quantize(torch.randn(E, N, K, device="cuda") * 0.05)
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    bias = torch.randn(N, device="cuda").to(torch.bfloat16)
    out = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)
    keep = None
    if where == "cpu":
        keep, wv = pinned_view(w8)
    else:
        wv = w8
    e = 1
    args = args_tensor(out=out.data_ptr(), ld_out=N, k_tiles=K // 64, epi=EPI_STORE, bm=bm,
                       n_limit=N, bias=bias.data_ptr(), w_fp8=1)
    wm, xm = weight_map_fp8(fab, wv), act_map(fab, x, bm)
    tiles = [Tile(KIND_GEMM, i=(n0, m0, min(bm, M - m0), e),
                  a=(args.data_ptr(), wm, xm, 0, sc[e].data_ptr()))
             for n0 in range(0, N, 128) for m0 in range(0, M, bm)]
    run_program(fab, tiles)
    ref = x.float() @ wd[e].t() + bias.float()
    assert (out.float() - ref).abs().max().item() < 0.02 * ref.abs().max().item()
    del keep


@pytest.mark.parametrize("form", ["sym", "asym"])
@pytest.mark.parametrize("where", ["hbm", "cpu"])
def test_expert_ffn_fp8(fab, form, where):
    """FP8 experts in both kernel forms: symmetric tiles with the SiLU
    epilogue, or asymmetric K-split tiles and their reduce."""
    from monoserve.fabric.args import device_block
    torch.manual_seed(4)
    E, H, I, T, TOPK = 4, 1024, 640, 48, 2
    w13_8, s13, w13d = quantize(torch.randn(E, 2 * I, H, device="cuda") * 0.03)
    w2_8, s2, w2d = quantize(torch.randn(E, H, I, device="cuda") * 0.03)
    x = torch.randn(T, H, device="cuda").to(torch.bfloat16)
    topk = torch.stack([torch.randperm(E, device="cuda")[:TOPK] for _ in range(T)])
    tw = torch.rand(T, TOPK, device="cuda")
    pairs = sorted((int(topk[t, k]), t, float(tw[t, k])) for t in range(T) for k in range(TOPK))
    xp = torch.stack([x[t] for _, t, _ in pairs])
    ptok = torch.tensor([t for _, t, _ in pairs], dtype=torch.int32, device="cuda")
    pw = torch.tensor([wt for _, _, wt in pairs], dtype=torch.float32, device="cuda")
    starts = {}
    for i, (e, _, _) in enumerate(pairs):
        starts.setdefault(e, [i, 0])[1] += 1
    ws = torch.zeros(len(pairs), 2 * I, device="cuda", dtype=torch.float32)
    hbuf = torch.zeros(len(pairs), I, device="cuda", dtype=torch.bfloat16)
    out32 = torch.zeros(T, H, device="cuda", dtype=torch.float32)
    keep = []
    if where == "cpu":
        k1, w13v = pinned_view(w13_8)
        k2, w2v = pinned_view(w2_8)
        keep = [k1, k2]
    else:
        w13v, w2v = w13_8, w2_8
    bm = 16
    m13, m2 = weight_map_fp8(fab, w13v), weight_map_fp8(fab, w2v)
    mx, mh = act_map(fab, xp, bm), act_map(fab, hbuf, bm)
    a2 = args_tensor(out=out32.data_ptr(), ld_out=H, k_tiles=I // 64, epi=EPI_WADD, bm=bm,
                     n_limit=H, pair_token=ptok.data_ptr(), pair_weight=pw.data_ptr(), w_fp8=1)
    p = Program()
    if form == "sym":
        a13 = args_tensor(out=hbuf.data_ptr(), ld_out=I, k_tiles=H // 64, epi=EPI_SILU_MUL, bm=bm,
                          n_limit=I, up_offset=I, w_fp8=1)
        t13, t2 = [], []
        for e, (s0, cnt) in starts.items():
            for m0 in range(s0, s0 + cnt, bm):
                mv = min(bm, s0 + cnt - m0)
                t13 += [Tile(KIND_GEMM, i=(f0, m0, mv, e),
                             a=(a13.data_ptr(), m13, mx, 0, s13[e].data_ptr())) for f0 in range(0, I, 64)]
                t2 += [Tile(KIND_GEMM, i=(n0, m0, mv, e),
                            a=(a2.data_ptr(), m2, mh, 0, s2[e].data_ptr())) for n0 in range(0, H, 128)]
        stages = [p.stage(t13), p.stage(t2)]
    else:
        a13 = args_tensor(out=ws.data_ptr(), ld_out=2 * I, k_tiles=H // 64, epi=EPI_ATOMIC, bm=bm,
                          n_limit=I, up_offset=I, w_fp8=1)
        red = device_block("ReduceArgs", ws=ws, out=hbuf, I=I, ld_ws=2 * I, ld_out=I)
        t13, tr, t2 = [], [], []
        for e, (s0, cnt) in starts.items():
            t13 += [Tile(KIND_GEMM_ASYM, i=(f0, s0, cnt, e),
                         a=(a13.data_ptr(), m13, mx, k0 | (kt << 16), s13[e].data_ptr()))
                    for f0 in range(0, I, 64) for k0, kt in k_splits(H // 64)]
            tr += [Tile(KIND_ASYM_REDUCE, i=(r0, min(64, s0 + cnt - r0)), a=(red.data_ptr(),))
                   for r0 in range(s0, s0 + cnt, 64)]
            t2 += [Tile(KIND_GEMM_ASYM, i=(n0, s0, cnt, e),
                        a=(a2.data_ptr(), m2, mh, k0 | (kt << 16), s2[e].data_ptr()))
                   for n0 in range(0, H, 128) for k0, kt in k_splits(I // 64)]
        stages = [p.stage(t13), p.stage(tr), p.stage(t2)]
    for a, b in zip(stages, stages[1:]):
        p.after(a, b)
    h = p.upload(fab, first=stages[0])
    fab.start()
    fab.publish(map=[0] * fab.num_workers, caps=[0], order=[0], programs=[h])
    import time
    t0 = time.time()
    while fab.progress()["lanes"][0]["iterations"] < 1:
        assert time.time() - t0 < 60
        time.sleep(1e-3)
    fab.stop()
    e_of = torch.tensor([e for e, _, _ in pairs], device="cuda")
    g = torch.einsum("mh,mih->mi", xp.float(), w13d[e_of, :I])
    u = torch.einsum("mh,mih->mi", xp.float(), w13d[e_of, I:])
    h_ref = torch.nn.functional.silu(g) * u
    assert (hbuf.float() - h_ref).abs().max().item() < 0.02 * h_ref.abs().max().item()
    y = torch.einsum("mi,mhi->mh", h_ref.to(torch.bfloat16).float(), w2d[e_of]) * pw[:, None]
    ref = torch.zeros(T, H, device="cuda").index_add_(0, ptok.long(), y)
    assert (out32 - ref).abs().max().item() < 0.03 * ref.abs().max().item()
    del keep


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("bm", [16, 64])
def test_dense_split_k(fab, bm, fp8):
    """A K-split GEMM: every tile takes one K range and adds its partial sums
    into an fp32 output, the tile of the first range with the bias."""
    torch.manual_seed(5)
    E, N, K, M = 2, 256, 1024, 40
    w = torch.randn(E, N, K, device="cuda") * 0.05
    x = torch.randn(M, K, device="cuda").to(torch.bfloat16)
    bias = torch.randn(N, device="cuda").to(torch.bfloat16)
    out = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    e = 1
    if fp8:
        wq, sc, wd = quantize(w)
        wm, scale = weight_map_fp8(fab, wq), sc[e].data_ptr()
    else:
        wq = w.to(torch.bfloat16)
        wd = wq.float()
        wm, scale = weight_map(fab, wq), 0
    args = args_tensor(out=out.data_ptr(), ld_out=N, k_tiles=K // 64, epi=EPI_ATOMIC, bm=bm,
                       n_limit=N, bias=bias.data_ptr(), w_fp8=int(fp8))
    xm = act_map(fab, x, bm)
    ranges = [(0, 5), (5, 5), (10, 6)]   # uneven ranges over the 16 K tiles
    tiles = [Tile(KIND_GEMM, i=(n0, m0, min(bm, M - m0), e),
                  a=(args.data_ptr(), wm, xm, k0 | (kn << 16), scale))
             for n0 in range(0, N, 128) for m0 in range(0, M, bm) for k0, kn in ranges]
    run_program(fab, tiles)
    ref = x.float() @ wd[e].t() + bias.float()
    assert (out - ref).abs().max().item() < 0.02 * ref.abs().max().item()
