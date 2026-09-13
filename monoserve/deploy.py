"""Build an engine for a checkpoint on this GPU.

HBM holds the non-expert weights, one staging buffer per prefill lane (two
full layers of experts each), the lanes' activation workspaces, the KV
cache, and, in what is left, the hot tier: per layer the experts with the
highest decode activation probability. The full expert set stays in pinned
CPU DRAM.
"""
import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from monoserve.engine import Engine, EngineConfig
from monoserve.placement import ActivationProfile, hot_tier
from monoserve.runtime.kinds import PAGE
from monoserve.runtime.lane import Lane

GB = 1e9


@dataclass
class DeployOptions:
    calibration: str                    # this GPU's calibration (python -m monoserve.calibrate)
    profile: str | None = None          # activation profiles (python -m monoserve.profile)
    kv_gb: float | None = None          # KV cache; default: four max_len requests
    hot_fraction: float | None = None   # hot tier as a share of the expert set; default: all HBM left
    hbm_budget_gb: float | None = None  # plan as if the GPU had this much HBM (default: what it has)
    headroom_gb: float = 2.0
    staging_layers: int = 2             # each staging buffer holds this many full layers
    prefill_lanes: int = 2
    token_budget: int = 16384
    max_batch_requests: int = 64
    max_decode: int = 256
    max_slots: int = 512
    max_len: int = 32768
    reserve_tokens: int = 1024
    alpha: float = 5.0
    forms: list = field(default_factory=lambda: [16, 32, 64, 128, 0])
    contention_aware: bool = True
    staging: bool = True
    engine: str = "full"                # ablations: "ml" (one mixed batch), "kf" (green contexts)

    @classmethod
    def from_dict(cls, d, **override):
        known = {k: v for k, v in {**d, **override}.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def load_profiles(path, cfg):
    """(decode ActivationProfile, prefill shares [layers, experts]); uniform
    without a profile file."""
    L, E = cfg.num_layers, cfg.num_experts
    decode = ActivationProfile(L, E)
    prefill = np.full((L, E), 1.0 / E)
    if path:
        with open(path) as f:
            data = json.load(f)
        decode.p = np.asarray(data["decode"], dtype=np.float64)
        prefill = np.asarray(data["prefill"], dtype=np.float64)
    return decode, prefill


def weight_bytes(cfg):
    return 1 if getattr(cfg, "fp8", False) else 2


def expert_bytes(cfg):
    return 3 * cfg.hidden * cfg.intermediate * weight_bytes(cfg)


def dense_bytes(cfg):
    """Non-expert weights: embeddings and the vocabulary projection (bf16),
    attention, routers, norms, shared experts, and dense layers' MLPs."""
    H, D, wb = cfg.hidden, cfg.head_dim, weight_bytes(cfg)
    moe_layers = cfg.num_layers - cfg.dense_layers
    attn = (H * cfg.qkv_dim + cfg.num_heads * D * H) * wb + 4 * H * 2
    per_moe = cfg.num_experts * H * 2 + 3 * H * cfg.shared_intermediate * wb
    per_dense = 3 * H * cfg.dense_intermediate * wb
    return (2 * cfg.vocab * H * 2 + cfg.num_layers * attn + moe_layers * per_moe
            + cfg.dense_layers * per_dense)


def plan_memory(cfg, opts, free_bytes):
    """Sizes of the HBM structures, in bytes, and the hot tier's size in
    experts. A hot tier capped by hot_fraction leaves its HBM to the KV
    cache, unless kv_gb fixes the cache."""
    w = expert_bytes(cfg)
    budget = min(free_bytes, opts.hbm_budget_gb * GB if opts.hbm_budget_gb else free_bytes)
    budget -= opts.headroom_gb * GB
    staging_experts = cfg.num_experts if opts.staging else 0   # V per half: one full layer
    halves = opts.staging_layers
    plan = {"dense": dense_bytes(cfg),
            "staging": opts.prefill_lanes * halves * staging_experts * w,
            "workspace": (2 * Lane.workspace_bytes(cfg, "decode", max_reqs=opts.max_decode,
                                                   max_len=opts.max_len)
                          + opts.prefill_lanes * Lane.workspace_bytes(
                              cfg, "prefill", max_tokens=opts.token_budget,
                              max_reqs=opts.max_batch_requests, max_len=opts.max_len))}
    page = PAGE * 2 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim * 2
    kv = opts.kv_gb * GB if opts.kv_gb else 4 * opts.max_len * page / PAGE
    left = budget - plan["dense"] - plan["staging"] - plan["workspace"] - kv
    all_experts = (cfg.num_layers - cfg.dense_layers) * cfg.num_experts
    hot = all_experts if opts.hot_fraction is None else int(opts.hot_fraction * all_experts)
    hot = max(0, min(hot, int(left // w)))
    if left < 0:
        raise MemoryError("the non-expert weights, staging buffers, workspaces, and KV cache "
                          "do not fit in HBM")
    if opts.hot_fraction is not None and not opts.kv_gb:
        kv += left - hot * w
    plan.update(kv=kv, kv_pages=int(kv // page), hot=hot * w, hot_experts=hot,
                staging_experts=staging_experts * halves // 2)
    return plan


def build_engine(model_path, opts, check_stops=True, log=print):
    import torch
    from transformers import AutoConfig

    from monoserve.runtime.checkpoint import load_checkpoint
    from monoserve.runtime.config import MoEConfig
    cfg = MoEConfig.from_hf(AutoConfig.from_pretrained(model_path))
    free, _ = torch.cuda.mem_get_info()
    decode_prof, prefill_p = load_profiles(opts.profile, cfg)
    plan = plan_memory(cfg, opts, free)
    # dense layers have no experts to keep
    ranked = SimpleNamespace(p=np.where(np.arange(cfg.num_layers)[:, None] < cfg.dense_layers, -1.0,
                                        decode_prof.p))
    hot = hot_tier(ranked, plan["hot_experts"])
    log("HBM plan (GB): " + ", ".join(f"{k} {plan[k] / GB:.1f}" for k in
                                       ("dense", "staging", "workspace", "kv", "hot")) +
        f"; hot tier {plan['hot_experts']} of "
        f"{(cfg.num_layers - cfg.dense_layers) * cfg.num_experts} experts")
    conf = EngineConfig(max_slots=opts.max_slots, kv_pages=plan["kv_pages"], max_len=opts.max_len,
                        token_budget=opts.token_budget, max_batch_requests=opts.max_batch_requests,
                        max_decode=opts.max_decode, prefill_lanes=opts.prefill_lanes,
                        staging_experts=plan["staging_experts"], reserve_tokens=opts.reserve_tokens,
                        alpha=opts.alpha, forms=tuple(opts.forms),
                        contention_aware=opts.contention_aware)

    def load(fab, **kw):
        return load_checkpoint(fab, model_path, hot, log=log, **kw)[1]

    from monoserve.ablations import GreenEngine, SingleLaneEngine
    engines = {"full": Engine, "ml": SingleLaneEngine, "kf": GreenEngine}
    if opts.engine not in engines:
        raise ValueError(f"unknown engine {opts.engine!r}: one of {sorted(engines)}")
    return engines[opts.engine](cfg, load, opts.calibration, conf, decode_profile=decode_prof,
                                prefill_profile=prefill_p, check_stops=check_stops, log=log)
