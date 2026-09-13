"""The contention-aware control plane (Section 3.1 of the paper).

The time estimator (Eq. 1 and 2), the plan search (Algorithm 1), and
admission are a C++ library (csrc/control), bound as monoserve._C.control.
This package re-exports it and adds calibration files, model shapes, and
activation-profile views.
"""
import numpy as np

from monoserve import _C
from monoserve.control.calibration import load_calibration, save_calibration

_ctl = _C.control
ASYM = _ctl.ASYM
Admission = _ctl.Admission
AdmissionConfig = _ctl.AdmissionConfig
Calibration = _ctl.Calibration
Curve = _ctl.Curve
Estimator = _ctl.Estimator
EstimatorOptions = _ctl.EstimatorOptions
KernelRates = _ctl.KernelRates
LaneWork = _ctl.LaneWork
ModelShape = _ctl.ModelShape
PlanSearch = _ctl.PlanSearch
ProfileView = _ctl.ProfileView
Request = _ctl.Request
SearchOptions = _ctl.SearchOptions
Seq = _ctl.Seq

# Kernel forms for DRAM experts: symmetric templates by tile height, and
# the asymmetric kernel.
FORMS = (16, 32, 64, 128, ASYM)

__all__ = ["ASYM", "FORMS", "Admission", "AdmissionConfig", "Calibration", "Curve",
           "Estimator", "EstimatorOptions", "KernelRates", "LaneWork", "ModelShape",
           "PlanSearch", "ProfileView", "Request", "SearchOptions", "Seq", "form_name",
           "load_calibration", "model_shape", "profile_view", "save_calibration"]


def form_name(form):
    return "asym" if form == ASYM else f"sym{form}"


def model_shape(cfg, expert_weight_bytes=None, dense_weight_bytes=None):
    """The estimator's ModelShape of a runtime MoEConfig (one byte per
    weight for FP8 checkpoints, two for bf16, unless given)."""
    wb = 1.0 if getattr(cfg, "fp8", False) else 2.0
    expert_weight_bytes = wb if expert_weight_bytes is None else expert_weight_bytes
    dense_weight_bytes = wb if dense_weight_bytes is None else dense_weight_bytes
    m = ModelShape()
    m.hidden, m.intermediate = cfg.hidden, cfg.intermediate
    m.experts, m.top_k, m.layers = cfg.num_experts, cfg.top_k, cfg.num_layers
    m.heads, m.kv_heads, m.head_dim = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    m.vocab = cfg.vocab
    m.dense_layers = getattr(cfg, "dense_layers", 0)
    m.dense_intermediate = getattr(cfg, "dense_intermediate", 0)
    m.shared_intermediate = getattr(cfg, "shared_intermediate", 0)
    m.expert_weight_bytes = float(expert_weight_bytes)
    m.dense_weight_bytes = float(dense_weight_bytes)
    return m


def profile_view(decode_p, prefill_p, hot):
    """A ProfileView from [layers, experts] activation shares of the decode
    and prefill profiles and the hot tier (a mask of the same shape, or one
    list of experts per layer)."""
    decode_p = np.asarray(decode_p, dtype=np.float64)
    layers, experts = decode_p.shape
    view = ProfileView(layers, experts)
    view.set_profile(True, decode_p.ravel().tolist())
    view.set_profile(False, np.asarray(prefill_p, dtype=np.float64).ravel().tolist())
    mask = np.zeros((layers, experts), dtype=np.uint8)
    if isinstance(hot, np.ndarray) and hot.shape == (layers, experts):
        mask[:] = hot != 0
    else:
        for layer, members in enumerate(hot):
            mask[layer, list(members)] = 1
    view.set_hot([int(v) for v in mask.ravel()])
    return view
