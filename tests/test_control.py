"""Tests of the control plane: the time estimator, the plan search, and
admission. They run on the CPU against a hand-written calibration whose
curves have the shapes the calibration run measures (rates rising with the
SM share and saturating, a link rate that grows with credits up to the
saturated rate, a residency stretch that grows with exposure)."""
import math

import numpy as np
import pytest

from monoserve.control import (ASYM, Admission, AdmissionConfig, Curve, Estimator,
                               EstimatorOptions, LaneWork, PlanSearch, Request, Seq,
                               load_calibration, model_shape, profile_view)
from monoserve.runtime.config import MoEConfig

S = 132
SMS = [8, 16, 24, 32, 48, 64, 96, 132]
R_C_US = 0.83
BETA = 400.0   # GB/s


def rates(tflops, gbps, knee, q):
    return {"sms": SMS, "tflops": [tflops * s / S for s in SMS],
            "gbps": [gbps * min(1.0, s / knee) for s in SMS], "q_per_sm": q}


def rho(eff):
    credits = [128, 256, 512, 1024, 2048, 4096, 8192, 10400, 12000]
    return {"credits": credits,
            "gbps": [min(BETA * eff, n * 32 / (R_C_US * 1e-6) / 1e9) for n in credits]}


CAL = {
    "sms": S, "R_H_us": 0.34, "R_C_us": R_C_US, "q_max": 96,
    "gamma": {"exposure_entry_us": [2000, 4000, 8000, 12000, 16000],
              "stretch": [1.0, 1.15, 1.5, 2.0, 2.5]},
    "dense": rates(600, 3600, 70, 64),
    "attn_prefill": rates(450, 3000, 80, 48),
    "attn_decode": rates(150, 3200, 64, 80),
    "expert": {"16": rates(160, 3400, 64, 96), "32": rates(300, 3400, 64, 96),
               "64": rates(480, 3400, 64, 96), "128": rates(620, 3400, 64, 96),
               "asym": rates(320, 3200, 64, 72)},
    "rho": {"16": rho(1.0), "32": rho(1.0), "64": rho(1.0), "128": rho(1.0),
            "asym": rho(0.8)},
    "q_form": {"16": 96, "32": 96, "64": 128, "128": 160, "asym": 64},
    "beta_gbps": BETA, "reference_form": 64,
    "layer_overhead_us": 4.0, "tail_overhead_us": 30.0,
}

QWEN = MoEConfig(hidden=2048, intermediate=768, num_experts=128, top_k=8, num_layers=48,
                 num_heads=32, num_kv_heads=4, vocab=151936)


def options(contention_aware):
    o = EstimatorOptions()
    o.contention_aware = contention_aware
    return o


@pytest.fixture(scope="module")
def cal():
    return load_calibration(CAL)


@pytest.fixture(scope="module")
def est(cal):
    return Estimator(model_shape(QWEN), cal)


def decode_work(batch=8, context=2000, active=27.0, miss=5.0, tpot=0.05):
    w = LaneWork()
    w.decode = True
    w.passes = [[Seq(1, context) for _ in range(batch)]]
    w.active, w.miss = [active] * 48, [miss] * 48
    w.deadline = tpot
    return w


def prefill_work(tokens=(4096,), deadline=2.0, staging=0, active=120.0, miss=30.0):
    w = LaneWork()
    w.passes = [[Seq(t, 0) for t in tokens]]
    w.active, w.miss = [active] * 48, [miss] * 48
    w.staging = staging
    w.deadline = deadline
    return w


def test_curve():
    c = Curve([8.0, 16.0], [80.0, 160.0], True)
    assert c(4) == pytest.approx(40) and c(12) == pytest.approx(120) and c(64) == pytest.approx(160)
    assert Curve([8.0, 16.0], [80.0, 160.0])(4) == pytest.approx(80)


def test_link_budget_and_exposure_bound(est, cal):
    # Q^link: the fewest credits at which the reference form saturates (to
    # 99%), read on the piecewise-linear curve between its grid points
    r = CAL["rho"]["64"]
    i = next(k for k, g in enumerate(r["gbps"]) if g >= 0.99 * BETA)
    x0, x1, y0, y1 = r["credits"][i - 1], r["credits"][i], r["gbps"][i - 1], r["gbps"][i]
    q_link = x0 + (0.99 * BETA - y0) / (y1 - y0) * (x1 - x0)
    assert est.link_credits == pytest.approx(q_link)
    assert est.exposure_bound == pytest.approx(q_link * 0.83e-6 + 96 * 132 * 0.34e-6)
    assert est.gamma_ratio(0.0) == pytest.approx(cal.gamma(est.exposure_bound))
    assert est.gamma_ratio(est.exposure_bound) == pytest.approx(1.0)
    assert Estimator(model_shape(QWEN), cal, options(False)).gamma_ratio(0.0) == 1.0


def test_estimate_falls_with_share(est):
    for w in (decode_work(), prefill_work()):
        load = est.load(w)
        t = [est.evaluate(load, s, est.link_credits, 64).total for s in range(1, S + 1)]
        assert all(b <= a * (1 + 1e-12) for a, b in zip(t, t[1:]))


def test_link_term(est, cal):
    load = est.load(decode_work(batch=8, active=27.0, miss=5.0))
    s, credits = 40, 2048.0
    e = est.evaluate(load, s, credits, 16)
    # a decode batch bounds M_e by its size: one pass of a 16-row template
    w = model_shape(QWEN).expert_bytes
    assert e.m_link == pytest.approx(5.0 * w / cal.rho_at(16, min(96 * s, credits)))


def test_staging_guarantee(est, cal):
    s, credits, form = 64, 4096.0, 64
    e = est.evaluate(est.load(prefill_work((8192,), staging=16)), s, credits, form)
    w = model_shape(QWEN).expert_bytes
    assert e.staged == pytest.approx(min(30.0, 16, math.floor(cal.rho_at(form, credits) * e.attn / w)))
    e0 = est.evaluate(est.load(prefill_work((8192,), staging=0)), s, credits, form)
    assert e0.staged == 0 and e.m_link < e0.m_link


def test_asymmetric_reads_each_expert_once(est, cal):
    # 2048 tokens * top-8 over 128 experts: 128 rows per expert
    load = est.load(prefill_work((2048,), active=128.0, miss=40.0))
    s, credits = 64, 8192.0
    sym, asym = est.evaluate(load, s, credits, 16), est.evaluate(load, s, credits, ASYM)
    b_sym = sym.m_link * cal.rho_at(16, min(96 * s, credits))
    b_asym = asym.m_link * cal.rho_at(ASYM, min(64 * s, credits))
    assert b_sym / b_asym == pytest.approx(8.0)


def test_deadline_floor_is_the_smallest_share(est):
    search = PlanSearch(est)
    for w in (decode_work(tpot=0.03), prefill_work(deadline=0.8)):
        load = est.load(w)
        for form in (16, 64, ASYM):
            brute = next((s for s in range(1, S + 1)
                          if est.evaluate(load, s, 4096.0, form).total <= w.deadline), S + 1)
            assert search.deadline_floor(load, 4096.0, form) == brute


def test_search_with_two_prefill_lanes(est):
    search = PlanSearch(est)
    lanes = [est.load(decode_work(batch=16, context=1000, tpot=0.05)),
             est.load(prefill_work((1000,) * 4, deadline=2.0, staging=32)),
             est.load(prefill_work((16384,), deadline=10.0, staging=32))]
    plan = search.solve(lanes, decode_key=7)
    assert plan.feasible and plan.min_slack >= 0
    assert sum(p.sms for p in plan.lanes) <= S
    assert sum(p.credits for p in plan.lanes) <= est.link_credits * 1.01 + 3 * 160
    assert sum(p.rate for p in plan.lanes) <= BETA * 1e9 * 1.01
    for p, load in zip(plan.lanes, lanes):
        assert p.est.total <= load.deadline and p.cap >= 1
    again = search.solve(lanes, decode_key=7)   # the decode lane's choices are cached
    assert again.feasible and again.evaluations < plan.evaluations


def test_search_reports_infeasible(est):
    plan = PlanSearch(est).solve([est.load(prefill_work((65536,), deadline=0.01))])
    assert not plan.feasible and len(plan.lanes) == 1 and plan.lanes[0].sms == S


def test_contention_blind_estimates_are_smaller(est, cal):
    mca = Estimator(model_shape(QWEN), cal, options(False))
    w = decode_work(batch=32, context=4000, tpot=0.03)
    for s in (16, 64, 132):
        assert mca.evaluate(mca.load(w), s, 1e9, 64).total <= est.evaluate(est.load(w), s, est.link_credits, 64).total
    f_ca = PlanSearch(est).deadline_floor(est.load(w), est.link_credits, 64)
    f_mca = PlanSearch(mca).deadline_floor(mca.load(w), mca.link_credits, 64)
    assert f_mca <= f_ca


def test_profile_view():
    p = np.full((2, 8), 1 / 8)
    view = profile_view(p, p, [[0, 1, 2, 3], []])
    a, m = view.expected(True, 0, 1.0)
    assert a == pytest.approx(1.0) and m == pytest.approx(0.5)
    a, m = view.expected(True, 0, 4096.0)
    assert a == pytest.approx(8, rel=1e-3) and m == pytest.approx(4, rel=1e-3)
    a, m = view.expected(False, 1, 16.0)
    assert m == pytest.approx(a) and a == pytest.approx(8 * (1 - (7 / 8) ** 16), rel=1e-9)


def make_admission(est, **cfg):
    c = AdmissionConfig()
    c.kv_bytes_per_token = model_shape(QWEN).kv_bytes_per_token
    for k, v in cfg.items():
        setattr(c, k, v)
    p = np.full((48, 128), 1 / 128)
    view = profile_view(p, p, [list(range(100))] * 48)
    return Admission(est, PlanSearch(est), view, c)


def test_admission_lanes_join_and_finish(est):
    adm = make_admission(est, token_budget=8192, staging_experts=32)
    d = adm.arrive(Request(1, 2000, 256, 0.0, 2.0, 0.05), 0.0)
    assert d.publish and d.feasible and d.pass_lanes == [1] and d.passes[0].reqs == [1]
    d = adm.arrive(Request(2, 3000, 256, 0.01, 2.0, 0.05), 0.01)
    assert d.pass_lanes == [2] and d.lanes == [1, 2]
    d = adm.arrive(Request(3, 1000, 256, 0.02, 2.0, 0.05), 0.02)
    assert not d.publish and adm.queued == 1        # both prefill lanes are busy
    d = adm.pass_done(1, 0.3)                       # request 1's first token is out
    assert d.joined == [1] and d.pass_lanes == [1] and d.passes[0].reqs == [3]
    assert d.lanes == [0, 2, 1] and d.feasible
    d = adm.finished(1, 0.5)
    assert d.publish and d.decode_changed and d.decode == []
    assert adm.kv_reserved == pytest.approx((3000 + 256 + 1000 + 256) * model_shape(QWEN).kv_bytes_per_token)


def test_admission_chunks_long_prompts(est):
    adm = make_admission(est, token_budget=4096)
    d = adm.arrive(Request(9, 10000, 16, 0.0, 30.0, 0.1), 0.0)
    assert d.pass_lanes == [1]
    p = d.passes[0]
    assert (p.tokens, p.pos0, p.last) == ([3334], [0], [0])
    d = adm.pass_done(1, 1.0)
    assert d.pass_lanes[0] == 1 and d.passes[0].pos0 == [3334]
    d = adm.pass_done(1, 2.0)
    assert d.passes[0].tokens == [3332] and d.passes[0].last == [1]
    d = adm.pass_done(1, 3.0)
    assert d.joined == [9]


def test_admission_rejects_only_hopeless_requests(est):
    kv = model_shape(QWEN).kv_bytes_per_token
    adm = make_admission(est, kv_capacity=kv * 3000)
    d = adm.arrive(Request(1, 65536, 16, 0.0, 1e-4, 0.05), 0.0)
    assert d.rejected == [1] and adm.queued == 0
    d = adm.arrive(Request(2, 2000, 256, 0.0, 5.0, 0.05), 0.0)
    assert d.pass_lanes == [1]
    # memory, not the deadline, holds this one back: it stays queued
    d = adm.arrive(Request(3, 2000, 256, 0.0, 5.0, 0.05), 0.0)
    assert d.pass_lanes == [] and d.rejected == [] and adm.queued == 1


def test_behind_republishes(est):
    adm = make_admission(est)
    adm.arrive(Request(1, 4000, 64, 0.0, 3.0, 0.05), 0.0)
    adm.progress(1, 20)
    d = adm.behind(1, 0.5)
    assert d.publish and d.lanes == [1] and d.searches == 1
