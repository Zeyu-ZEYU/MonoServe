"""Tests of the SLO metrics and targets of the evaluation harness (CPU only,
no server): TTFT and TPOT of records, attainment with unfinished requests,
nearest-rank P99, goodput interpolation, and the 1K TTFT floor."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import metrics  # noqa: E402
from eval.solo import Target, bucket_profile, targets  # noqa: E402
from eval.workloads import REFERENCE_ID  # noqa: E402


def rec(rid="a", send=1.0, chunk_t=(1.5, 1.6, 1.7, 1.8), chunk_n=None,
        n_out=None, error=None):
    chunk_n = list(chunk_n or [1] * len(chunk_t))
    return {"id": rid, "arrival": send, "send": send,
            "first": chunk_t[0] if chunk_t else None,
            "finish": None if error else (chunk_t[-1] if chunk_t else send),
            "chunk_t": list(chunk_t), "chunk_n": chunk_n,
            "output_tokens": sum(chunk_n) if n_out is None else n_out,
            "error": error}


def test_ttft_and_tpot():
    r = rec()
    assert metrics.ttft(r) == pytest.approx(0.5)
    assert metrics.tpot(r) == pytest.approx(0.1)
    # one chunk carrying four tokens: TPOT spreads the span over n_out - 1
    r = rec(chunk_t=(1.5, 1.9), chunk_n=(1, 4))
    assert metrics.tpot(r) == pytest.approx(0.4 / 4)


def test_tpot_needs_two_tokens():
    r = rec(chunk_t=(1.2,))
    assert metrics.tpot(r) is None
    t = Target(0.1, 0.5, 0.01)
    assert metrics.slo_met(r, t)                       # TTFT 0.2 <= 0.5
    assert not metrics.slo_met(rec(chunk_t=(1.7,)), t)  # TTFT 0.7


def test_attainment_counts_unfinished_as_violations():
    t = {k: Target(0.1, 0.6, 0.15) for k in "abcd"}
    recs = [rec("a"), rec("b"),
            rec("c", chunk_t=(1.8, 1.9)),                # TTFT 0.8 > 0.6
            rec("d", error="unfinished")]
    s = metrics.summarize(recs, t)
    assert s["attainment"] == pytest.approx(0.5)
    assert s["finished"] == 3
    # the TPOT side: 0.1 s per token against a 0.05 s target
    t2 = {k: Target(0.1, 0.6, 0.05) for k in "ab"}
    assert metrics.summarize([rec("a"), rec("b")], t2)["attainment"] == 0


def test_p99_nearest_rank():
    vals = list(range(1, 101))
    assert metrics.percentile(vals, 99) == 99
    assert metrics.percentile(vals, 50) == 50
    assert metrics.percentile(vals[:-1] + [math.inf], 99) == 99
    assert metrics.percentile(vals[:-2] + [math.inf] * 2, 99) == math.inf
    assert metrics.percentile([], 99) is None


def test_p99_tails_go_infinite_past_one_percent_unfinished():
    t = {str(i): Target(0.25, 1.25, 0.5) for i in range(100)}
    recs = [rec(str(i)) for i in range(100)]
    s = metrics.summarize(recs, t)
    assert s["p99_ttft_ratio"] == pytest.approx(2.0)      # 0.5 s / 0.25 s
    assert s["p99_tpot_ms"] == pytest.approx(100.0)
    recs[0]["error"] = recs[1]["error"] = "unfinished"
    s = metrics.summarize(recs, t)
    assert s["p99_ttft_ratio"] == math.inf and s["p99_tpot_ms"] == math.inf


def test_goodput_interpolation():
    rates, att = [1, 2, 4, 8], [1.0, 0.95, 0.85, 0.2]
    assert metrics.goodput(rates, att) == pytest.approx(3.0)
    assert metrics.goodput(rates[::-1], att[::-1]) == pytest.approx(3.0)
    # the first drop counts, not a later recovery
    assert metrics.goodput([1, 2, 3], [1.0, 0.8, 0.95]) == pytest.approx(1.5)
    assert metrics.goodput([1, 2], [0.85, 0.5]) is None
    assert metrics.goodput([1, 2], [0.99, 0.9]) == 2
    assert metrics.censored([0.99, 0.9]) and not metrics.censored(att)
    assert metrics.goodput(rates, att, level=0.5) == pytest.approx(
        4 + (0.85 - 0.5) / (0.85 - 0.2) * 4)


def solo_entries():
    return {"solo": {
        REFERENCE_ID: {"prompt_tokens": 1024, "ttft": 0.2, "tpot": 0.02},
        "short": {"prompt_tokens": 32, "ttft": 0.05, "tpot": 0.010},
        "long": {"prompt_tokens": 8000, "ttft": 1.0, "tpot": 0.012},
        "one": {"prompt_tokens": 600, "ttft": 0.1, "tpot": None},
    }}


def test_targets_take_the_1k_floor():
    reqs = [{"id": k, "prompt_tokens": n} for k, n in
            (("short", 32), ("long", 8000), ("one", 600))]
    t = targets(reqs, solo_entries(), alpha=5)
    assert t["short"].ttft_ref == pytest.approx(0.2)      # floored at 1K
    assert t["short"].ttft == pytest.approx(1.0)
    assert t["short"].tpot == pytest.approx(0.05)
    assert t["long"].ttft == pytest.approx(5.0)
    assert t["long"].tpot == pytest.approx(0.06)
    # no solo TPOT: the median of the measured requests (0.011 s)
    assert t["one"].tpot == pytest.approx(5 * 0.011)
    assert targets(reqs, solo_entries(), alpha=2)["long"].ttft == \
        pytest.approx(2.0)


def test_targets_need_every_solo_run_in_exact_mode():
    with pytest.raises(KeyError):
        targets([{"id": "missing", "prompt_tokens": 10}], solo_entries())


def test_bucket_mode_interpolates_over_prompt_length():
    solo = solo_entries()
    (x, y), (xp, yp) = bucket_profile(
        [e for k, e in solo["solo"].items() if k != REFERENCE_ID])
    # 32 and 600 share the first bucket (median 316 tokens, 0.075 s)
    assert x == [316, 8000] and y == pytest.approx([0.075, 1.0])
    assert xp == [316, 8000] and yp == pytest.approx([0.010, 0.012])
    reqs = [{"id": "mid", "prompt_tokens": 4300},
            {"id": "huge", "prompt_tokens": 100000}]
    t = targets(reqs, solo, alpha=1, mode="buckets")
    assert t["mid"].ttft == pytest.approx(
        0.075 + (4300 - 316) / (8000 - 316) * (1.0 - 0.075))
    assert t["huge"].ttft == pytest.approx(1.0)           # flat beyond
    assert t["huge"].tpot == pytest.approx(0.012)
