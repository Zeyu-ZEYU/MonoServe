"""Model weights in the fabric's layout.

Dense weights (embeddings, attention projections, norms, routers, the
vocabulary projection, a shared expert, the MLPs of dense layers) live in
HBM. The full expert set of every MoE layer lives in pinned CPU DRAM, where
GEMM tiles read it over the link through TMA. A per-layer hot tier keeps
chosen experts in HBM, and each prefill lane owns a staging buffer the copy
engine fills. Each layer has an indirection table that maps every expert to
(region, slot); the expander reads it when it turns routed experts into
tiles, so moving an expert between regions is a copy followed by one table
write.

Weights are bf16, or FP8 (e4m3, stored as raw bytes) with one fp32 scale per
output feature; the scales stay in HBM, indexed by expert, wherever the
expert's weights are.
"""
import weakref

import numpy as np
import torch

from monoserve import _C
from monoserve.runtime.kinds import REGION_HOST, REGION_HOT, REGION_STAGE0

FP8_DTYPES = (torch.uint8, torch.float8_e4m3fn)


def is_fp8(t):
    return t is not None and t.dtype in FP8_DTYPES


def weight_map(fab, t, pinned=False):
    """TMA map of a [Z, rows, K] weight (Z experts or slots), bf16 or FP8."""
    if t.dim() == 2:
        t = t.unsqueeze(0)
    z, rows, k = t.shape
    base = _C.host_device_pointer(t.data_ptr()) if pinned else t.data_ptr()
    if is_fp8(t):
        return fab.tensor_map(base, [k, rows, z], [k, rows * k], [64, 64, 1], dtype=2, swizzle=0)
    return fab.tensor_map(base, [k, rows, z], [k * 2, rows * k * 2], [64, 64, 1])


def as_stored(t):
    """A weight as the fabric stores it: FP8 as raw bytes, fp32 kept,
    everything else bf16."""
    if t is None:
        return None
    if is_fp8(t):
        return t.contiguous().view(torch.uint8)
    if t.dtype == torch.float32:
        return t.contiguous()
    return t.to(torch.bfloat16).contiguous()


def to_dev(t):
    t = as_stored(t)
    return None if t is None else t.to("cuda")


def pin(t):
    return as_stored(t).pin_memory()


class ModelWeights:
    def __init__(self, fab, cfg, dense, experts, hot, staging_slots=0, staging_buffers=2,
                 expert_scales=None):
        """dense: dict with embed [V, H], lm_head [V, H], final_norm [H] and
        per-layer lists in_norm, qkv [Nqkv, H], qkv_bias, q_norm, k_norm,
        o [H, hq*D], post_norm, router [E, H], and optionally router_bias
        [E] (fp32, sigmoid routing), shared_w13 [2Is, H] and shared_w2
        [H, Is] (a shared expert), mlp_w13 [2Id, H] and mlp_w2 [H, Id] (a
        dense layer's MLP), and the fp32 per-row scales of FP8 weights as
        <name>_scale. Missing entries are None.
        experts: per layer (w13 [E, 2I, H], w2 [E, H, I]), None for a dense
        layer; expert_scales: per layer (s13 [E, 2I], s2 [E, H]) for FP8.
        hot: per layer, the experts kept in HBM. All device memory is
        allocated here, before the fabric starts."""
        self.fab, self.cfg = fab, cfg
        self.lanes = weakref.WeakSet()   # lanes with their own tables (no cycle)
        L, E = cfg.num_layers, cfg.num_experts

        def per_layer(name):
            v = dense.get(name)
            return [to_dev(t) for t in v] if v is not None else [None] * L

        self.embed = to_dev(dense["embed"])
        self.lm_head = to_dev(dense["lm_head"])
        self.final_norm = to_dev(dense["final_norm"])
        for name in ("in_norm", "qkv", "qkv_bias", "q_norm", "k_norm", "o", "post_norm", "router",
                     "router_bias", "shared_w13", "shared_w2", "mlp_w13", "mlp_w2", "qkv_scale",
                     "o_scale", "shared_w13_scale", "shared_w2_scale", "mlp_w13_scale",
                     "mlp_w2_scale"):
            setattr(self, name, per_layer(name))
        maps = lambda ws: [None if w is None else weight_map(fab, w) for w in ws]  # noqa: E731
        self.qkv_map, self.o_map, self.router_map = maps(self.qkv), maps(self.o), maps(self.router)
        self.shared_w13_map, self.shared_w2_map = maps(self.shared_w13), maps(self.shared_w2)
        self.mlp_w13_map, self.mlp_w2_map = maps(self.mlp_w13), maps(self.mlp_w2)
        self.lm_map = weight_map(fab, self.lm_head)

        self.host_w13 = [None if x is None else pin(x[0]) for x in experts]
        self.host_w2 = [None if x is None else pin(x[1]) for x in experts]
        scales = expert_scales or [None] * L
        self.expert_s13 = [None if s is None else to_dev(s[0].float().reshape(E, -1)) for s in scales]
        self.expert_s2 = [None if s is None else to_dev(s[1].float().reshape(E, -1)) for s in scales]
        self.fp8_experts = any(is_fp8(w) for w in self.host_w13)
        self.hot = [list(h) if experts[layer] is not None else [] for layer, h in enumerate(hot)]
        self.hot_w13, self.hot_w2 = [], []
        for layer in range(L):
            if self.host_w13[layer] is None:
                self.hot_w13.append(None)
                self.hot_w2.append(None)
                continue
            ids = self.hot[layer] or [0]
            self.hot_w13.append(self.host_w13[layer][ids].to("cuda"))
            self.hot_w2.append(self.host_w2[layer][ids].to("cuda"))
        I, H = cfg.intermediate, cfg.hidden
        edt = torch.uint8 if self.fp8_experts else torch.bfloat16
        slots = max(1, staging_slots)
        self.staging_slots = staging_slots
        self.stage_w13 = [torch.zeros(slots, 2 * I, H, dtype=edt, device="cuda")
                          for _ in range(staging_buffers)]
        self.stage_w2 = [torch.zeros(slots, H, I, dtype=edt, device="cuda")
                         for _ in range(staging_buffers)]
        stage13 = [weight_map(fab, t) for t in self.stage_w13]
        stage2 = [weight_map(fab, t) for t in self.stage_w2]
        self.n_regions = REGION_STAGE0 + staging_buffers
        self.w13_maps, self.w2_maps, self.table, self.table_host = [], [], [], []
        for layer in range(L):
            tab = np.stack([np.full(E, REGION_HOST, np.int32), np.arange(E, dtype=np.int32)], 1)
            if self.host_w13[layer] is None:   # a dense layer: no experts
                self.w13_maps.append(None)
                self.w2_maps.append(None)
            else:
                m13 = [weight_map(fab, self.host_w13[layer], pinned=True),
                       weight_map(fab, self.hot_w13[layer])] + stage13
                m2 = [weight_map(fab, self.host_w2[layer], pinned=True),
                      weight_map(fab, self.hot_w2[layer])] + stage2
                self.w13_maps.append(torch.tensor(m13, dtype=torch.int64, device="cuda"))
                self.w2_maps.append(torch.tensor(m2, dtype=torch.int64, device="cuda"))
                for slot, e in enumerate(self.hot[layer]):
                    tab[e] = (REGION_HOT, slot)
            self.table_host.append(tab)
            self.table.append(torch.from_numpy(tab.copy()).to("cuda"))

    def expert_bytes(self):
        """(w13, w2) bytes of one expert as stored."""
        w = next(x for x in self.host_w13 if x is not None)
        v = next(x for x in self.host_w2 if x is not None)
        return w[0].numel() * w.element_size(), v[0].numel() * v.element_size()

    def set_location(self, layer, expert, region, slot):
        """Point an expert at another region in the base table and in every
        lane's table; safe while the fabric runs."""
        self.table_host[layer][expert] = (region, slot)
        row = np.array([region, slot], dtype=np.int32).tobytes()
        self.fab.write(self.table[layer].data_ptr() + 8 * expert, row)
        for lane in self.lanes:
            self.fab.write(lane.tables[layer].data_ptr() + 8 * expert, row)

    @classmethod
    def synthetic(cls, fab, cfg, hot, seed=0, scale=0.05, **kw):
        """Random weights of the given shape, for tests and smoke runs. With
        cfg.fp8, every linear weight except the router and the vocabulary
        projection is FP8 with per-row scales, like an FP8 checkpoint."""
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: (torch.randn(*s, generator=g) * scale).to(torch.bfloat16)  # noqa: E731
        ones = lambda n: (1 + 0.1 * torch.randn(n, generator=g)).to(torch.bfloat16)  # noqa: E731
        L, H, I, E, D = cfg.num_layers, cfg.hidden, cfg.intermediate, cfg.num_experts, cfg.head_dim
        dl = cfg.dense_layers
        dense = {
            "embed": r(cfg.vocab, H) * 20, "lm_head": r(cfg.vocab, H), "final_norm": ones(H),
            "in_norm": [ones(H) for _ in range(L)], "qkv": [r(cfg.qkv_dim, H) for _ in range(L)],
            "qkv_bias": [r(cfg.qkv_dim) * 4 if cfg.qkv_bias else None for _ in range(L)],
            "q_norm": [ones(D) if cfg.qk_norm else None for _ in range(L)],
            "k_norm": [ones(D) if cfg.qk_norm else None for _ in range(L)],
            "o": [r(H, cfg.num_heads * D) for _ in range(L)],
            "post_norm": [ones(H) for _ in range(L)],
            "router": [r(E, H) * 4 if l >= dl else None for l in range(L)],
            "router_bias": [(torch.randn(E, generator=g) * 0.1).float()
                            if cfg.scoring == "sigmoid" and l >= dl else None for l in range(L)],
            "shared_w13": [r(2 * cfg.shared_intermediate, H) if cfg.shared_intermediate and l >= dl
                           else None for l in range(L)],
            "shared_w2": [r(H, cfg.shared_intermediate) if cfg.shared_intermediate and l >= dl
                          else None for l in range(L)],
            "mlp_w13": [r(2 * cfg.dense_intermediate, H) if l < dl else None for l in range(L)],
            "mlp_w2": [r(H, cfg.dense_intermediate) if l < dl else None for l in range(L)],
        }
        experts = [(r(E, 2 * I, H), r(E, H, I)) if l >= dl else None for l in range(L)]
        scales = None
        if cfg.fp8:
            for name in ("qkv", "o", "shared_w13", "shared_w2", "mlp_w13", "mlp_w2"):
                qs = [quantize_fp8(w) for w in dense[name]]
                dense[name] = [None if q is None else q[0] for q in qs]
                dense[name + "_scale"] = [None if q is None else q[1] for q in qs]
            scales = []
            for layer, x in enumerate(experts):
                if x is None:
                    scales.append(None)
                    continue
                (q13, s13), (q2, s2) = quantize_fp8(x[0]), quantize_fp8(x[1])
                experts[layer] = (q13, q2)
                scales.append((s13, s2))
        return cls(fab, cfg, dense, experts, hot, expert_scales=scales, **kw)

    def dequantized(self, name, layer):
        """A dense weight in fp32, scales applied (for references)."""
        w, s = getattr(self, name)[layer], getattr(self, name + "_scale")[layer]
        return fp8_to_float(w, s) if is_fp8(w) else w.float()

    def expert_weights(self, layer):
        """(w13, w2) of a layer in fp32 on the GPU, scales applied."""
        w13, w2 = self.host_w13[layer].cuda(), self.host_w2[layer].cuda()
        if is_fp8(w13):
            return (fp8_to_float(w13, self.expert_s13[layer]), fp8_to_float(w2, self.expert_s2[layer]))
        return w13.float(), w2.float()


def quantize_fp8(w):
    """Per-row FP8 e4m3 quantization: (e4m3 weights, fp32 scale per row)."""
    if w is None:
        return None
    x = w.float()
    s = (x.abs().amax(dim=-1, keepdim=True) / 448.0).clamp_min(1e-12)
    return (x / s).to(torch.float8_e4m3fn), s.squeeze(-1)


def fp8_to_float(w, s):
    q = w.view(torch.float8_e4m3fn) if w.dtype == torch.uint8 else w
    return q.float() * s.float().to(q.device).unsqueeze(-1)
