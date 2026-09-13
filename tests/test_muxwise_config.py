"""MuxWise's division table (CPU): loading, the spread divisions, the
group choice, and the table built from profile measurements."""
import json

import pytest

from monoserve.baselines.muxwise.config import MuxConfig, divide_sm, dump, load, select_group
from monoserve.baselines.muxwise.profile import candidate_divisions, table

ROWS = [[112, 20, 1], [104, 28, 5], [96, 36, 10], [80, 52, 15], [64, 68, 20], [56, 76, 25]]


def test_load_and_dump(tmp_path):
    cfg = MuxConfig(sm_group_num=8, manual_divisions=ROWS)
    dump(cfg, str(tmp_path / "d.yaml"))
    back = load(str(tmp_path / "d.yaml"))
    assert back.manual_divisions == ROWS and back.sm_group_num == 8
    (tmp_path / "bad.json").write_text(json.dumps({"sm_group_num": 5, "manual_divisions": ROWS}))
    with pytest.raises(ValueError):
        load(str(tmp_path / "bad.json"))
    assert load(None) == MuxConfig()


def test_select_group():
    cfg = MuxConfig(sm_group_num=8, manual_divisions=ROWS)
    assert select_group(cfg, 0, True, False) == 0          # prefill alone
    assert select_group(cfg, 7, False, True) == 7          # decode alone
    assert select_group(cfg, 1, True, True) == 1
    assert select_group(cfg, 12, True, True) == 3          # last threshold reached: 10
    assert select_group(cfg, 400, True, True) == 6
    high = MuxConfig(sm_group_num=4, manual_divisions=[[96, 36, 4], [64, 68, 16]])
    assert select_group(high, 2, True, True) == 1          # below every threshold
    spread = MuxConfig(sm_group_num=8)
    assert 1 <= select_group(spread, 10, True, True) <= 6


def test_divide_sm():
    divs = divide_sm(132, (9, 0), 6)
    assert all(p % 8 == 0 and p + d == 132 and d >= 16 and p >= d for p, d in divs)
    assert divs == sorted(divs, key=lambda x: -x[0])
    cands = candidate_divisions(132)
    assert all(p % 8 == 0 and p + d == 132 for p, d in cands)
    assert [d for _, d in cands] == sorted(d for _, d in cands)


def test_table_from_measurements():
    sms = [(132, 0), (112, 20), (96, 36), (64, 68), (0, 132)]
    bss, toks = [1, 8, 64], [1024]
    solo_full = {1: 0.010, 8: 0.011, 64: 0.020}
    # decode alone on each division's decode share, and beside a prefill
    solo = {1: {1: 0.020, 8: 0.030, 64: 0.200}, 2: {1: 0.015, 8: 0.018, 64: 0.060},
            3: {1: 0.012, 8: 0.013, 64: 0.030}}
    data = {"sms": sms, "solo_full": solo_full,
            "solo": {f"{g}:{b}": solo[g][b] for g in solo for b in bss},
            "corun": {f"{g}:{b}:1024": 1.5 * solo[g][b] for g in solo for b in bss}}
    rows, target = table(data, bss, toks, alpha=5.0)
    assert target == pytest.approx(0.05)
    # bs 1 and 8 fit the smallest decode share (0.03, 0.045 <= 0.05); bs 64
    # needs 68 SMs, from right after the largest size the 20 SMs served
    assert rows == [[112, 20, 1], [64, 68, 9]]
