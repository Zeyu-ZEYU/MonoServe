"""The calibration run on a small model shape: every section of the
calibration file is produced from fabric runs, has the expected shape, and
loads into the time estimator."""
import numpy as np
import pytest
import torch

from monoserve.runtime.config import MoEConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

CFG = MoEConfig(hidden=512, intermediate=256, num_experts=16, top_k=4, num_layers=2,
                num_heads=4, num_kv_heads=2, vocab=1024)


def non_decreasing(v):
    return all(b >= a for a, b in zip(v, v[1:]))


def test_quick_calibration():
    from monoserve.calibrate import Calibrator
    from monoserve.control import (Estimator, LaneWork, PlanSearch, Seq, load_calibration,
                                   model_shape)
    data = Calibrator(CFG, quick=True, log=lambda *a: None).run()
    assert 0 < data["R_H_us"] < data["R_C_us"]
    for name in ("dense", "attn_prefill", "attn_decode"):
        r = data[name]
        assert len(r["sms"]) == len(r["tflops"]) == len(r["gbps"])
        assert min(r["tflops"]) > 0 and min(r["gbps"]) > 0 and r["q_per_sm"] > 0
        assert non_decreasing(r["tflops"]) and non_decreasing(r["gbps"])
    assert set(data["expert"]) == set(data["rho"]) == {"16", "32", "64", "128", "asym"}
    for form, r in data["rho"].items():
        assert non_decreasing(r["gbps"]) and all(np.diff(r["credits"]) > 0), form
        assert data["q_form"][form] > 0
    g = data["gamma"]
    assert g["stretch"][0] == 1.0 and non_decreasing(g["stretch"])
    assert all(np.diff(g["exposure_entry_us"]) > 0)
    assert data["beta_gbps"] > 0 and data["layer_overhead_us"] > 0

    est = Estimator(model_shape(CFG), load_calibration(data))
    w = LaneWork()
    w.decode = True
    w.passes = [[Seq(1, 500) for _ in range(4)]]
    w.active, w.miss = [8.0] * 2, [3.0] * 2
    w.deadline = 1.0
    load = est.load(w)
    t = est.evaluate(load, data["sms"], est.link_credits, 64).total
    assert 0 < t < 1.0
    assert PlanSearch(est).deadline_floor(load, est.link_credits, 64) <= data["sms"]
