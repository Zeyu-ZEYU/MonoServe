"""Lanes and their programs: a MoE transformer as stages of tiles.

A lane is one batch in flight on the fabric. Its program chains, per layer,
the attention window (QKV projection, q/k norm and RoPE with the KV-cache
append, attention, output projection) and the MoE window (norm, router,
expansion into expert tiles, permutation, expert w13 and w2 tiles, residual
combination with the next layer's norm). A prefill program runs once over
its batch's prompt chunk and samples each finished prompt's first token; a
decode program advances every request of its batch by one token per
iteration and repeats until the host publishes a newer program.

All device memory of a lane is allocated at construction, before the fabric
starts; building a program afterwards only writes through the fabric's
staged copies and its program pool, which is safe while the kernel runs.
"""
import math

import numpy as np

from monoserve.fabric.args import layouts, pack
from monoserve.fabric.program import Program, tile_array
from monoserve.runtime import kinds as K
from monoserve.runtime.kinds import HEAD_DIM, PAGE, TILE_HEIGHTS, tile_height
from monoserve.runtime.weights import is_fp8

ROW_TILE = 16
ROPE_ITEMS = 48       # (row, head) items per rotary tile: four per warp
SPLIT_MIN_KT = 4      # K tiles (of 64) a tile of a K-split GEMM covers at least
_ROPE = {}


def _ptr(t):
    return 0 if t is None else t.data_ptr()


def rope_table(theta, rot, max_pos):
    """fp32 [max_pos, rot]: the cos of the rot/2 rotary angles of every
    position, then their sin (computed like the reference, shared by the
    lanes)."""
    import torch
    key = (float(theta), int(rot), int(max_pos))
    if key not in _ROPE:
        half = rot // 2
        j = torch.arange(half, dtype=torch.float32, device="cuda")
        inv = 1.0 / torch.pow(torch.tensor(theta, dtype=torch.float32, device="cuda"), 2 * j / rot)
        ang = torch.arange(max_pos, device="cuda")[:, None].float() * inv[None]
        _ROPE[key] = torch.cat([ang.cos(), ang.sin()], dim=1).contiguous()
    return _ROPE[key]


def row_tiles(kind, args, rows, per=ROW_TILE):
    i0 = np.arange(0, rows, per)
    return tile_array(kind, i0=i0, i1=np.minimum(per, rows - i0), a0=args)


def gemm_tiles(args, wmap, xmap, n, m, h, z=0, step=128, scale=0, k_tiles=0, splits=1):
    """GEMM tiles over n output features (step per tile: 128, or 64 for
    the gate-and-up tiles of a w13) and m activation rows in blocks of h;
    scale: the fp32 per-row scales of an FP8 weight. With splits > 1, the
    k_tiles K tiles are cut into that many even ranges, one per tile, and
    the GEMM's epilogue must add (fp32 atomics or the weighted add)."""
    n0, m0 = np.meshgrid(np.arange(0, n, step), np.arange(0, m, h), indexing="ij")
    n0, m0 = n0.ravel(), m0.ravel()
    a3 = 0
    if splits > 1:
        per = -(-k_tiles // splits)
        ranges = np.asarray([k0 | (min(per, k_tiles - k0) << 16) for k0 in range(0, k_tiles, per)],
                            dtype=np.int64)
        a3 = np.tile(ranges, len(n0))
        n0, m0 = np.repeat(n0, len(ranges)), np.repeat(m0, len(ranges))
    return tile_array(K.GEMM, i0=n0, i1=m0, i2=np.minimum(h, m - m0), i3=z,
                      a0=args, a1=wmap, a2=xmap, a3=a3, a4=scale)


def act_map(fab, t2d, box_rows):
    rows, k = t2d.shape
    return fab.tensor_map(t2d.data_ptr(), [k, rows], [t2d.stride(0) * 2], [64, box_rows])


class Lane:
    def __init__(self, fab, cfg, weights, kv, requests, lane_id, mode, max_tokens=0,
                 max_reqs=8, max_len=None, chunk_pages=16, seed=0):
        import torch
        assert mode in ("prefill", "decode")
        self.fab, self.cfg, self.w, self.kv, self.req = fab, cfg, weights, kv, requests
        self.lane_id, self.mode = lane_id, mode
        H, I, E, Kt, V = cfg.hidden, cfg.intermediate, cfg.num_experts, cfg.top_k, cfg.vocab
        hq, hkv, D = cfg.num_heads, cfg.num_kv_heads, HEAD_DIM
        T = max_tokens if mode == "prefill" else max_reqs
        R = max_reqs
        self.T, self.R, self.P = T, R, T * Kt
        self.max_len = max_len or requests.bt_stride * PAGE
        self.chunk_pages = chunk_pages
        self.chunk_keys = chunk_pages * PAGE
        self.max_chunks = -(-self.max_len // self.chunk_keys)
        bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32

        def z(*shape, dt=bf):
            return torch.zeros(*shape, dtype=dt, device="cuda")

        self.x, self.x2, self.xn = z(T, H), z(T, H), z(T, H)
        self.qkv = z(T, cfg.qkv_dim)
        self.q, self.ao = z(T, hq, D), z(T, hq, D)
        self.rl = z(T, E, dt=f32)
        self.topk_ids, self.topk_w = z(T, Kt, dt=i32), z(T, Kt, dt=f32)
        self.counts, self.fill, self.starts = z(E, dt=i32), z(E, dt=i32), z(E + 1, dt=i32)
        self.xp, self.h = z(self.P, H), z(self.P, I)
        self.pair_token, self.pair_w = z(self.P, dt=i32), z(self.P, dt=f32)
        self.acc = z(T, H, dt=f32)
        self.ws_asym = z(self.P, 2 * I, dt=f32)   # w13 partial sums of the asymmetric form
        # a shared expert and dense layers' MLPs add into acc row for row
        Is, Id = cfg.shared_intermediate, cfg.dense_intermediate if cfg.dense_layers else 0
        self.hs = z(T, Is) if Is else None
        self.hm = z(T, Id) if Id else None
        self.ident = torch.arange(T, dtype=i32, device="cuda")
        self.unit = torch.ones(T, dtype=f32, device="cuda")
        self.tokens, self.tok_pos, self.tok_slot = z(T, dt=i32), z(T, dt=i32), z(T, dt=i32)
        self.req_info = z(max(T, R), 4, dt=i32)
        self.gather = torch.full((T,), -1, dtype=i32, device="cuda")
        self.row_req = z(R, dt=i32)
        self.bt = z(R, requests.bt_stride, dt=i32)
        self.xlast, self.logits = z(R, H), z(R, V, dt=f32)
        if mode == "prefill":
            self.ws_o, self.ws_m, self.ws_l = z(T, hq, D, dt=f32), z(T, hq, dt=f32), z(T, hq, dt=f32)
            self.q_map = fab.tensor_map(self.q.data_ptr(), [64, T, 2, hq, 1],
                                        [hq * D * 2, 128, D * 2, T * hq * D * 2], [64, 64, 2, 1, 1])
        else:
            self.ws_o = z(R, hq, self.max_chunks, D, dt=f32)
            self.ws_m = z(R, hq, self.max_chunks, dt=f32)
            self.ws_l = self.ws_m
            self.q_map = fab.tensor_map(self.q.data_ptr(), [64, T * hq, 2, 1, 1],
                                        [D * 2, 128, T * hq * D * 2, T * hq * D * 2], [64, 64, 2, 1, 1])

        # This lane's indirection tables, [layers][experts][2]: its own copy,
        # so staging flips of one prefill lane never reach another lane.
        self.tables = torch.stack(weights.table).clone()
        weights.lanes.add(self)
        # rows routed to each expert, per layer, accumulated by the expander
        self.hist = z(cfg.num_layers, E, dt=i32)

        self.xn_maps = {h: act_map(fab, self.xn, h) for h in TILE_HEIGHTS}
        self.ao_maps = {h: act_map(fab, self.ao.view(T, hq * D), h) for h in TILE_HEIGHTS}
        self.xlast_maps = {h: act_map(fab, self.xlast, h) for h in TILE_HEIGHTS}
        self.hs_maps = {h: act_map(fab, self.hs, h) for h in TILE_HEIGHTS} if Is else None
        self.hm_maps = {h: act_map(fab, self.hm, h) for h in TILE_HEIGHTS} if Id else None
        xp_maps = [act_map(fab, self.xp, h) for h in TILE_HEIGHTS]
        h_maps = [act_map(fab, self.h, h) for h in TILE_HEIGHTS]
        self.xp_maps_dev = torch.tensor(xp_maps, dtype=torch.int64, device="cuda")
        self.h_maps_dev = torch.tensor(h_maps, dtype=torch.int64, device="cuda")

        blob = self._blob
        L = cfg.num_layers
        self.qkv_args = [{h: blob("GemmArgs", out=self.qkv, ld_out=cfg.qkv_dim, k_tiles=H // 64,
                                  epi=K.EPI_STORE, bm=h, n_limit=cfg.qkv_dim,
                                  bias=_ptr(weights.qkv_bias[l]), w_fp8=int(is_fp8(weights.qkv[l])))
                          for h in TILE_HEIGHTS} for l in range(L)]
        self.o_args = {h: blob("GemmArgs", out=self.x2, ld_out=H, k_tiles=hq * D // 64,
                               epi=K.EPI_STORE, bm=h, n_limit=H, residual=self.x, ld_res=H,
                               w_fp8=int(is_fp8(weights.o[0])))
                       for h in TILE_HEIGHTS}
        # The router GEMM adds (K-split) sums into the logits, which the
        # router zeroes once read.
        self.rg_args = {h: blob("GemmArgs", out=self.rl, ld_out=E, k_tiles=H // 64,
                                epi=K.EPI_ATOMIC_F32, bm=h, n_limit=E) for h in TILE_HEIGHTS}
        self.lm_args = {h: blob("GemmArgs", out=self.logits, ld_out=V, k_tiles=H // 64,
                                epi=K.EPI_F32, bm=h, n_limit=V) for h in TILE_HEIGHTS}
        f8 = int(weights.fp8_experts)
        w13 = [blob("GemmArgs", out=self.h, ld_out=I, k_tiles=H // 64, epi=K.EPI_SILU_MUL, bm=h,
                    n_limit=I, up_offset=I, w_fp8=f8) for h in TILE_HEIGHTS]
        w2 = [blob("GemmArgs", out=self.acc, ld_out=H, k_tiles=I // 64, epi=K.EPI_WADD, bm=h,
                   n_limit=H, pair_token=self.pair_token, pair_weight=self.pair_w, w_fp8=f8)
              for h in TILE_HEIGHTS]
        self.w13_args_dev = torch.tensor(w13, dtype=torch.int64, device="cuda")
        self.w2_args_dev = torch.tensor(w2, dtype=torch.int64, device="cuda")
        # The asymmetric form (K-split tiles over all rows of a DRAM expert,
        # row blocks of at most 64): w13 partial sums go to ws_asym and a
        # reduce tile applies silu(gate) * up; w2 adds into the accumulator.
        asym_h = [min(h, 64) for h in TILE_HEIGHTS]
        w13a = [blob("GemmArgs", out=self.ws_asym, ld_out=2 * I, k_tiles=H // 64,
                     epi=K.EPI_ATOMIC_F32, bm=h, n_limit=I, up_offset=I, w_fp8=f8) for h in asym_h]
        w2a = [blob("GemmArgs", out=self.acc, ld_out=H, k_tiles=I // 64, epi=K.EPI_WADD, bm=h,
                    n_limit=H, pair_token=self.pair_token, pair_weight=self.pair_w, w_fp8=f8)
               for h in asym_h]
        self.w13_asym_dev = torch.tensor(w13a, dtype=torch.int64, device="cuda")
        self.w2_asym_dev = torch.tensor(w2a, dtype=torch.int64, device="cuda")
        self.red_args = blob("ReduceArgs", ws=self.ws_asym, out=self.h, I=I, ld_ws=2 * I, ld_out=I)

        def ffn_args(width, out, ws):
            """A dense FFN: w13 with the SiLU-and-multiply epilogue into
            `out`, w2 added row for row into the MoE accumulator."""
            fp8 = int(any(is_fp8(x) for x in ws if x is not None))
            a13 = {h: blob("GemmArgs", out=out, ld_out=width, k_tiles=H // 64,
                           epi=K.EPI_SILU_MUL, bm=h, n_limit=width, up_offset=width, w_fp8=fp8)
                   for h in TILE_HEIGHTS}
            a2 = {h: blob("GemmArgs", out=self.acc, ld_out=H, k_tiles=width // 64, epi=K.EPI_WADD,
                          bm=h, n_limit=H, pair_token=self.ident, pair_weight=self.unit, w_fp8=fp8)
                  for h in TILE_HEIGHTS}
            return a13, a2
        if Is:
            self.sh13_args, self.sh2_args = ffn_args(Is, self.hs, weights.shared_w13)
        if Id:
            self.mlp13_args, self.mlp2_args = ffn_args(Id, self.hm, weights.mlp_w13)

        step = 0
        if mode == "decode":
            step = blob("StepArgs", row_req=self.row_req, last_token=requests.last_token,
                        seq_len=requests.seq_len, block_table=requests.block_table,
                        tok_pos=self.tok_pos, tok_slot=self.tok_slot, req_info=self.req_info,
                        bt_stride=requests.bt_stride)
        eps = cfg.rms_eps
        self.embed_args = blob("NormArgs", x=self.x, w=weights.in_norm[0], out=self.xn,
                               embed=weights.embed, tokens=self.tokens, step=step, H=H, ld=H, eps=eps)
        self.post_args = [blob("NormArgs", x=self.x2, w=weights.post_norm[l], out=self.xn, H=H,
                               ld=H, eps=eps) for l in range(L)]
        self.addnorm_args = [
            blob("NormArgs", x=self.x, x2=self.x2, acc=self.acc,
                 w=weights.in_norm[l + 1] if l + 1 < L else weights.final_norm, out=self.xn,
                 gather=self.gather if l + 1 == L else 0, out2=self.xlast, H=H, ld=H, eps=eps)
            for l in range(L)]
        max_pos = max(self.max_len, requests.bt_stride * PAGE)
        self.rope = rope_table(cfg.rope_theta, cfg.rotary_dim, max_pos)
        self.qk_heads = hq + 2 * hkv
        self.qk_args = [blob("QkArgs", qkv=self.qkv, q_out=self.q, k_cache=kv.k[l], v_cache=kv.v[l],
                             q_norm=_ptr(weights.q_norm[l]), k_norm=_ptr(weights.k_norm[l]),
                             tok_pos=self.tok_pos, tok_slot=self.tok_slot, ld_qkv=cfg.qkv_dim,
                             hq=hq, hkv=hkv, rot_dim=cfg.rotary_dim, theta=float(cfg.rope_theta),
                             eps=eps, cos_sin=self.rope.data_ptr(), max_pos=max_pos)
                        for l in range(L)]
        self.router_args = [blob("RouterArgs", logits=self.rl, topk_ids=self.topk_ids,
                                 topk_w=self.topk_w, counts=self.counts,
                                 bias=_ptr(weights.router_bias[l]), E=E, K=Kt,
                                 scoring=0 if cfg.scoring == "softmax" else 1,
                                 renorm=int(cfg.renormalize), scale=float(cfg.routed_scale))
                            for l in range(L)]
        self.permute_args = blob("PermuteArgs", topk_ids=self.topk_ids, topk_w=self.topk_w,
                                 starts=self.starts, fill=self.fill, pair_token=self.pair_token,
                                 pair_weight=self.pair_w, src=self.xn, dst=self.xp, K=Kt, H=H)
        scale_log2 = (1.0 / math.sqrt(D)) * math.log2(math.e)
        self.attn_args = [blob("AttnArgs", q_map=self.q_map, k_map=kv.k_maps[l], v_map=kv.v_maps[l],
                               out=self.ao, ws_o=self.ws_o, ws_m=self.ws_m, ws_l=self.ws_l,
                               block_table=self.bt, req_info=self.req_info, scale_log2=scale_log2,
                               bt_stride=requests.bt_stride, hq=hq, hkv=hkv, group=hq // hkv,
                               chunk_blocks=chunk_pages, max_chunks=self.max_chunks)
                          for l in range(L)]
        self.sample_args = blob("SampleArgs", logits=self.logits, temperature=requests.temperature,
                                last_token=requests.last_token, steps=requests.steps,
                                host_tokens=requests.host_tokens_dev,
                                host_steps=requests.host_steps_dev, seed=seed, V=V,
                                ring=requests.ring)
        self.programs = {}
        self.labels = {}      # handle -> stage labels, for stage traces

    @staticmethod
    def workspace_bytes(cfg, mode, max_tokens=0, max_reqs=8, max_len=32768, chunk_pages=16):
        """HBM a lane of this shape allocates (its activation workspace)."""
        H, I, E, Kt, V = cfg.hidden, cfg.intermediate, cfg.num_experts, cfg.top_k, cfg.vocab
        hq, D = cfg.num_heads, HEAD_DIM
        T = max_tokens if mode == "prefill" else max_reqs
        R, P = max_reqs, T * Kt
        chunks = -(-max_len // (chunk_pages * PAGE))
        b = 3 * T * H * 2 + T * cfg.qkv_dim * 2 + 2 * T * hq * D * 2 + T * E * 4 + 2 * T * Kt * 4
        b += P * H * 2 + P * I * 2 + 2 * P * 4 + T * H * 4 + P * 2 * I * 4
        b += 4 * T * 4 + max(T, R) * 16 + R * 4 + R * -(-max_len // PAGE) * 4
        b += R * H * 2 + R * V * 4
        b += T * (cfg.shared_intermediate + (cfg.dense_intermediate if cfg.dense_layers else 0)) * 2 + T * 8
        if mode == "prefill":
            b += T * hq * D * 4 + 2 * T * hq * 4
        else:
            b += R * hq * chunks * (D + 1) * 4 + cfg.num_layers * E * 4
        return b

    def _splits(self, n, rows, h, k_tiles, step=128):
        """K ranges for a GEMM whose tiles alone would leave most workers idle
        (fewer tiles than an eighth of the workers): enough for a tile per
        worker, each range SPLIT_MIN_KT K tiles or more."""
        tiles = -(-n // step) * -(-rows // h)
        if 8 * tiles >= self.fab.num_workers:
            return 1
        return max(1, min(-(-self.fab.num_workers // tiles), k_tiles // SPLIT_MIN_KT))

    # ------------------------------------------------------------------
    def _blob(self, name, **fields):
        data = pack(name, **fields)
        addr = self.fab.blob_alloc(len(data))
        self.fab.write(addr, data)
        return addr

    def _write(self, tensor, array, rows_offset=0):
        arr = np.ascontiguousarray(array)
        row_bytes = tensor.element_size() * (tensor[0].numel() if tensor.dim() > 1 else 1)
        self.fab.write(tensor.data_ptr() + rows_offset * row_bytes, arr.tobytes())

    def _expert_bound(self, rows):
        """Most expert tiles one layer can generate, under any kernel form."""
        cfg = self.cfg
        pairs = rows * cfg.top_k
        E, H, I = cfg.num_experts, cfg.hidden, cfg.intermediate
        sym = (-(-pairs // 16) + E) * (I // 64 + H // 128)
        splits = lambda kt: -(-kt // K.ASYM_MAX_KT)  # noqa: E731
        asym = E * ((I // 64) * splits(H // 64) + (H // 128) * splits(I // 64)) + -(-pairs // 64) + E
        return sym + asym

    def _layers(self, p, prev, rows, attention, blobs, pending):
        cfg, w = self.cfg, self.w
        H, E = cfg.hidden, cfg.num_experts
        h = tile_height(rows)
        ex_size = layouts()["ExpandArgs"][0]
        kq = H // 64
        for l in range(cfg.num_layers):
            s_qkv = p.stage(gemm_tiles(self.qkv_args[l][h], w.qkv_map[l], self.xn_maps[h],
                                       cfg.qkv_dim, rows, h, scale=_ptr(w.qkv_scale[l])),
                            label=("qkv", l))
            p.after(prev, s_qkv)
            s_rope = p.stage(row_tiles(K.QK_ROPE, self.qk_args[l], rows * self.qk_heads,
                                       per=ROPE_ITEMS), mark=(l << 8) | 1, label=("rope", l))
            p.after(s_qkv, s_rope)
            s_att = attention(p, l, s_rope, blobs, pending)
            s_o = p.stage(gemm_tiles(self.o_args[h], w.o_map[l], self.ao_maps[h], H, rows, h,
                                     scale=_ptr(w.o_scale[l])), mark=(l << 8) | 2,
                          label=("o", l))
            p.after(s_att, s_o)
            s_n2 = p.stage(row_tiles(K.RMSNORM, self.post_args[l], rows), label=("norm", l))
            p.after(s_o, s_n2)
            if l < cfg.dense_layers:
                # a dense layer: its MLP in place of the MoE window
                kd = cfg.dense_intermediate // 64
                s_m13 = p.stage(gemm_tiles(self.mlp13_args[h], w.mlp_w13_map[l], self.xn_maps[h],
                                           cfg.dense_intermediate, rows, h, step=64,
                                           scale=_ptr(w.mlp_w13_scale[l])), label=("mlp13", l))
                p.after(s_n2, s_m13)
                s_m2 = p.stage(gemm_tiles(self.mlp2_args[h], w.mlp_w2_map[l], self.hm_maps[h], H,
                                          rows, h, scale=_ptr(w.mlp_w2_scale[l]), k_tiles=kd,
                                          splits=self._splits(H, rows, h, kd)), label=("mlp2", l))
                p.after(s_m13, s_m2)
                s_an = p.stage(row_tiles(K.ADD_NORM, self.addnorm_args[l], rows), mark=(l << 8) | 3,
                               label=("add_norm", l))
                p.after(s_m2, s_an)
                prev = s_an
                continue
            s_rg = p.stage(gemm_tiles(self.rg_args[h], w.router_map[l], self.xn_maps[h], E, rows, h,
                                      k_tiles=kq, splits=self._splits(E, rows, h, kq)),
                           label=("router_gemm", l))
            p.after(s_n2, s_rg)
            s_rt = p.stage(row_tiles(K.ROUTER, self.router_args[l], rows), label=("router", l))
            p.after(s_rg, s_rt)
            ex = self.fab.blob_alloc(ex_size)
            blobs.append(ex)
            s_ex = p.stage(tile_array(K.EXPAND, a0=ex), label=("expand", l))
            p.after(s_rt, s_ex)
            s_pm = p.stage(row_tiles(K.PERMUTE, self.permute_args, rows), label=("permute", l))
            p.after(s_ex, s_pm)
            # per expert: w13, the asymmetric form's reduce (empty under a
            # symmetric form), w2; the expander fills them at run time
            w13 = [p.stage(label=("w13", l)) for _ in range(E)]
            red = [p.stage(label=("red", l)) for _ in range(E)]
            w2 = [p.stage(label=("w2", l)) for _ in range(E)]
            s_an = p.stage(row_tiles(K.ADD_NORM, self.addnorm_args[l], rows), mark=(l << 8) | 3,
                           label=("add_norm", l))
            for e in range(E):
                p.after(s_pm, w13[e])
                p.after(w13[e], red[e])
                p.after(red[e], w2[e])
                p.after(w2[e], s_an)
            if cfg.shared_intermediate:
                # the shared expert runs beside the routed ones
                s_s13 = p.stage(gemm_tiles(self.sh13_args[h], w.shared_w13_map[l], self.xn_maps[h],
                                           cfg.shared_intermediate, rows, h, step=64,
                                           scale=_ptr(w.shared_w13_scale[l])),
                                label=("shared13", l))
                p.after(s_n2, s_s13)
                ks = cfg.shared_intermediate // 64
                s_s2 = p.stage(gemm_tiles(self.sh2_args[h], w.shared_w2_map[l], self.hs_maps[h], H,
                                          rows, h, scale=_ptr(w.shared_w2_scale[l]), k_tiles=ks,
                                          splits=self._splits(H, rows, h, ks)),
                               label=("shared2", l))
                p.after(s_s13, s_s2)
                p.after(s_s2, s_an)
            pending.append(("expand", ex, l, w13[0], w2[0], red[0], s_an))
            prev = s_an
        return prev

    def _finish(self, p, first, iterations, blobs, pending, dyn, rows):
        cfg, w = self.cfg, self.w
        p.reserve(dyn)
        dyn_first = p.dynamic_first
        for item in pending:
            if item[0] == "expand":
                _, addr, l, s13, s2, sred, join = item
                self.fab.write(addr, pack(
                    "ExpandArgs", counts=self.counts, starts=self.starts, fill=self.fill,
                    table=self.tables[l], w13_maps=w.w13_maps[l], w2_maps=w.w2_maps[l],
                    hist=0 if self.hist is None else self.hist[l].data_ptr(),
                    xp_maps=self.xp_maps_dev, h_maps=self.h_maps_dev, w13_args=self.w13_args_dev,
                    w2_args=self.w2_args_dev, w13_asym_args=self.w13_asym_dev,
                    w2_asym_args=self.w2_asym_dev, red_args=self.red_args,
                    w13_scale=_ptr(w.expert_s13[l]), w2_scale=_ptr(w.expert_s2[l]), E=cfg.num_experts,
                    H=cfg.hidden, I=cfg.intermediate, host_region=K.REGION_HOST,
                    w13_stage0=s13, w2_stage0=s2, red_stage0=sred, join_stage=join,
                    dyn_first=dyn_first, dyn_cap=dyn, lane=self.lane_id))
            else:
                _, addr, l, s_att, s_cmb = item
                self.fab.write(addr, pack(
                    "AttnPlanArgs", attn=self.attn_args[l], req_info=self.req_info, rows=rows,
                    hkv=cfg.num_kv_heads, chunk_keys=self.chunk_keys, stage_attn=s_att,
                    stage_comb=s_cmb, dyn_first=dyn_first, dyn_cap=dyn))
        handle = p.upload(self.fab, first=first, iterations=iterations)
        self.programs[handle] = blobs
        self.labels[handle] = p.labels()
        return handle

    def _sample_stages(self, p, prev, samples):
        rows = len(samples)
        h = tile_height(rows)
        s_lm = p.stage(gemm_tiles(self.lm_args[h], self.w.lm_map, self.xlast_maps[h],
                                  self.cfg.vocab, rows, h), label=("lm_head", -1))
        p.after(prev, s_lm)
        s_sp = p.stage(tile_array(K.SAMPLE, i0=np.arange(rows), i1=np.asarray(samples),
                                  a0=self.sample_args), label=("sample", -1))
        p.after(s_lm, s_sp)
        return s_sp

    # ------------------------------------------------------------------
    def build_prefill(self, reqs):
        """reqs: dicts with slot, tokens (this chunk), pos0 (position of the
        chunk's first token), pages (the request's KV pages), and last
        (whether the prompt ends in this chunk, so its first token is
        sampled). Returns the program handle."""
        assert self.mode == "prefill"
        cfg = self.cfg
        stride = self.req.bt_stride
        tokens, pos, slots, info, gather, bt, samples = [], [], [], [], [], [], []
        rows = 0
        for r in reqs:
            n, p0, pages = len(r["tokens"]), int(r["pos0"]), list(r["pages"])
            ps = np.arange(p0, p0 + n)
            tokens.append(np.asarray(r["tokens"], dtype=np.int32))
            pos.append(ps)
            pg = np.asarray(pages, dtype=np.int64)
            slots.append((pg[ps // PAGE] * PAGE + ps % PAGE).astype(np.int32))
            info.append([rows, p0, n, p0 + n])
            g = np.full(n, -1, dtype=np.int32)
            if r.get("last", True):
                g[-1] = len(samples)
                samples.append(int(r["slot"]))
            gather.append(g)
            bt.append(pages + [0] * (stride - len(pages)))
            rows += n
        if rows > self.T or len(reqs) > self.R:
            raise ValueError("prefill batch exceeds the lane's buffers")
        self._write(self.tokens, np.concatenate(tokens))
        self._write(self.tok_pos, np.concatenate(pos).astype(np.int32))
        self._write(self.tok_slot, np.concatenate(slots))
        self._write(self.req_info, np.asarray(info, dtype=np.int32))
        self._write(self.gather, np.concatenate(gather))
        self._write(self.bt, np.asarray(bt, dtype=np.int32))

        by_chunk = {}
        for ri, r in enumerate(reqs):
            n, p0 = len(r["tokens"]), int(r["pos0"])
            kv_len = p0 + n
            for qb in range(-(-n // 128)):
                keys_end = min(kv_len, p0 + min(n, qb * 128 + 128))
                for c in range(-(-keys_end // self.chunk_keys)):
                    by_chunk.setdefault(c, []).append((ri, qb, c))
        hq = cfg.num_heads

        def attention(p, l, prev, blobs, pending):
            for c in sorted(by_chunk):
                arr = np.asarray(by_chunk[c])
                ri = np.repeat(arr[:, 0], hq)
                qb = np.repeat(arr[:, 1], hq)
                hh = np.tile(np.arange(hq), len(arr))
                s = p.stage(tile_array(K.ATTN_PREFILL, i0=ri, i1=hh, i2=qb, i3=c,
                                       a0=self.attn_args[l]), label=("attn", l))
                p.after(prev, s)
                prev = s
            return prev

        p = Program()
        blobs, pending = [], []
        s_emb = p.stage(row_tiles(K.EMBED, self.embed_args, rows), label=("embed", -1))
        last = self._layers(p, s_emb, rows, attention, blobs, pending)
        if samples:
            self._sample_stages(p, last, samples)
        return self._finish(p, s_emb, 1, blobs, pending, self._expert_bound(rows), rows)

    def build_decode(self, slots, pages, iterations=0):
        """Decode program over request slots (pages[slot]: the request's KV
        pages, covering its prompt and output limit)."""
        assert self.mode == "decode"
        B = len(slots)
        if B > self.R:
            raise ValueError("decode batch exceeds the lane's buffers")
        stride = self.req.bt_stride
        self._write(self.row_req, np.asarray(slots, dtype=np.int32))
        self._write(self.gather, np.arange(B, dtype=np.int32))
        self._write(self.bt, np.asarray([list(pages[s]) + [0] * (stride - len(pages[s]))
                                         for s in slots], dtype=np.int32))
        plan_size = layouts()["AttnPlanArgs"][0]

        def attention(p, l, prev, blobs, pending):
            plan = self.fab.blob_alloc(plan_size)
            blobs.append(plan)
            s_plan = p.stage(tile_array(K.ATTN_PLAN, a0=plan), label=("attn_plan", l))
            p.after(prev, s_plan)
            s_att = p.stage(label=("attn", l))
            p.after(s_plan, s_att)
            s_cmb = p.stage(label=("attn_combine", l))
            p.after(s_att, s_cmb)
            pending.append(("plan", plan, l, s_att, s_cmb))
            return s_cmb

        p = Program()
        blobs, pending = [], []
        s_emb = p.stage(row_tiles(K.EMBED, self.embed_args, B), label=("embed", -1))
        last = self._layers(p, s_emb, B, attention, blobs, pending)
        self._sample_stages(p, last, list(slots))
        attn_bound = B * self.cfg.num_kv_heads * self.max_chunks + B
        dyn = max(self._expert_bound(B), attn_bound)
        return self._finish(p, s_emb, iterations, blobs, pending, dyn, B)

    def release(self, handle):
        """Free a program the lane no longer runs."""
        self.fab.release(handle)
        self.labels.pop(handle, None)
        for b in self.programs.pop(handle, []):
            self.fab.blob_free(b)

    def program_labels(self, handle):
        """Stage labels of a program this lane built (monoserve.fabric.trace)."""
        return self.labels.get(handle)
