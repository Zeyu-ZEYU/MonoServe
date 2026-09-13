"""Mechanism tests of the MonoFab runtime with test tiles.

They check that every tile runs exactly once and never before the stages
it depends on, that workers serve their own lane first and borrow idle
capacity from other lanes, that link slots bound the DRAM-expert tiles in
flight, that a new epoch reaches every worker within about one tile, and
that programs iterate and are replaced at iteration boundaries.

No other CUDA work may run while the fabric occupies every SM, so each test
allocates its buffers before start() and reads them after stop().
"""
import time

import numpy as np
import pytest
import torch

from monoserve.fabric import KIND, Fabric, Program, Tile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs a CUDA GPU")

TRACE_WORDS = 4   # TraceRec: start, end, (worker, sm), (count, lane)


def trace_buffer(n):
    return torch.zeros(n, TRACE_WORDS, dtype=torch.int64, device="cuda")


def decode_trace(t):
    t = t.cpu().numpy().astype(np.uint64)
    return {
        "start": t[:, 0].astype(np.int64),
        "end": t[:, 1].astype(np.int64),
        "worker": (t[:, 2] & 0xFFFFFFFF).astype(np.int64),
        "count": (t[:, 3] & 0xFFFFFFFF).astype(np.int64),
        "lane": (t[:, 3] >> np.uint64(32)).astype(np.int64),
    }


def spin(ns, trace, idx, lane_tag=0):
    return Tile(KIND["spin"], i=(idx, lane_tag), a=(ns, trace.data_ptr()))


def wait_idle(fab, lane, timeout=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if fab.lane_stats(lane)["running"] == 0 and fab.progress()["lanes"][lane]["iterations"] > 0:
            return
        time.sleep(0.01)
    raise TimeoutError(f"lane {lane} did not finish")


def all_lane(fab, lane):
    return [lane] * fab.num_workers


@pytest.fixture
def fab():
    """A fabric that is stopped even when a test fails: a running fabric
    holds every SM, so a leaked one would stall every later test."""
    f = Fabric()
    yield f
    f.stop()


def wait_until(pred, timeout=30.0, what="condition"):
    t0 = time.time()
    while not pred():
        if time.time() - t0 > timeout:
            raise TimeoutError(what)
        time.sleep(1e-4)


def test_dag_order_and_exactly_once(fab):
    trace = trace_buffer(4096)
    p = Program()
    idx = iter(range(4096))
    s0 = p.stage([spin(20_000, trace, next(idx)) for _ in range(40)])
    s1 = p.stage([spin(15_000, trace, next(idx)) for _ in range(300)])
    s2 = p.stage([spin(30_000, trace, next(idx)) for _ in range(100)])
    s3 = p.stage([spin(10_000, trace, next(idx)) for _ in range(64)])
    p.after(s0, s1)
    p.after(s0, s2)
    p.after(s1, s3)
    p.after(s2, s3)
    h = p.upload(fab, first=s0)
    fab.start()
    fab.publish(map=all_lane(fab, 0), caps=[0], order=[0], programs=[h])
    wait_idle(fab, 0)
    fab.stop()
    tr = decode_trace(trace)
    bounds = {s0: (0, 40), s1: (40, 340), s2: (340, 440), s3: (440, 504)}
    assert (tr["count"][:504] == 1).all(), "every tile runs exactly once"
    for pred, succ in ((s0, s1), (s0, s2), (s1, s3), (s2, s3)):
        a, b = bounds[pred]
        c, d = bounds[succ]
        assert tr["start"][c:d].min() >= tr["end"][a:b].max()
    assert len(set(tr["worker"][40:340])) > fab.num_workers // 2, "work spreads"


def test_floors_and_borrowing(fab):
    n = fab.num_workers
    half = n // 2
    trace = trace_buffer(40_000)
    # Lane 0 has a queue that lasts ~10 ms on half the SMs; lane 1 has a
    # short one, so its workers borrow lane 0 tiles before and after it.
    p0, p1 = Program(), Program()
    p0.stage([spin(20_000, trace, i, 0) for i in range(30_000)])
    p1.stage([spin(20_000, trace, 32_768 + i, 1) for i in range(200)])
    h0 = p0.upload(fab, first=0)
    h1 = p1.upload(fab, first=0)
    lane_of = [0] * half + [1] * (n - half)
    fab.start()
    # Lane 0 starts first and keeps its workers busy; then lane 1 arrives.
    # (When both start at once, workers whose lane is not activated yet
    # borrow the other lane's tiles for one tile, as intended.)
    fab.publish(map=lane_of, caps=[0, 0], order=[0, 1], programs=[h0, 0])
    time.sleep(0.001)
    fab.publish(map=lane_of, caps=[0, 0], order=[0, 1], programs=[h0, h1])
    wait_idle(fab, 0)
    wait_idle(fab, 1)
    fab.stop()
    tr = decode_trace(trace)
    assert (tr["count"][:30_000] == 1).all() and (tr["count"][32_768:32_968] == 1).all()
    w0 = tr["worker"][:30_000]
    w1 = tr["worker"][32_768:32_968]
    own1 = np.isin(w1, np.arange(half, n)).mean()
    borrowed0 = np.isin(w0, np.arange(half, n)).mean()
    assert own1 > 0.9, f"lane 1 tiles should run on lane 1 workers ({own1:.2f})"
    assert borrowed0 > 0.2, f"idle lane 1 workers should borrow lane 0 tiles ({borrowed0:.2f})"


@pytest.mark.parametrize("cap", [2, 8])
def test_link_slots_bound_inflight(fab, cap):
    src = torch.randint(0, 255, (256 << 20,), dtype=torch.uint8, device="cuda")
    sink = torch.zeros(fab.num_workers, dtype=torch.int32, device="cuda")
    trace = trace_buffer(2048)
    chunk = 1 << 20
    p = Program()
    link = p.stage([Tile(KIND["read"], i=(i,), a=(src.data_ptr() + (i % 256) * chunk,
                                                   chunk, sink.data_ptr(), trace.data_ptr()))
                    for i in range(1024)], link=True)
    other = p.stage([spin(5_000, trace, 1100 + i) for i in range(400)])
    p.after(link, other)
    h = p.upload(fab, first=link)
    fab.start()
    fab.reset_inflight_max(0)
    fab.publish(map=all_lane(fab, 0), caps=[cap], order=[0], programs=[h])
    wait_idle(fab, 0)
    stats = fab.lane_stats(0)
    fab.stop()
    tr = decode_trace(trace)
    assert (tr["count"][:1024] == 1).all()
    assert (tr["count"][1100:1500] == 1).all()
    assert stats["inflight_max"] <= cap
    assert stats["inflight_max"] == cap, "the gate should admit exactly cap tiles"


def test_epoch_reaches_workers_within_a_tile(fab):
    trace = trace_buffer(1 << 14)
    tile_ns = 50_000
    p = Program()
    p.stage([spin(tile_ns, trace, i) for i in range(12_000)])
    h = p.upload(fab, first=0)
    fab.start()
    fab.publish(map=all_lane(fab, 0), caps=[0], order=[0], programs=[h])
    time.sleep(0.2)
    t_host = time.perf_counter()
    e = fab.publish(map=all_lane(fab, 0), caps=[0], order=[0], programs=[h])
    wait_until(lambda: min(fab.progress()["worker_epoch"]) >= e, what="epoch")
    elapsed_us = (time.perf_counter() - t_host) * 1e6
    ns = np.array(fab.progress()["worker_epoch_ns"], dtype=np.int64)
    fab.stop()
    spread_us = (ns.max() - ns.min()) / 1e3
    # every worker sees the new epoch at its next tile boundary
    assert spread_us <= tile_ns / 1e3 * 1.5, spread_us
    print(f"epoch published -> all {fab.num_workers} workers: {elapsed_us:.0f} us; "
          f"device-side spread {spread_us:.1f} us for {tile_ns / 1e3:.0f} us tiles")


def test_iterations(fab):
    trace = trace_buffer(1024)
    p = Program()
    a = p.stage([spin(2_000, trace, i) for i in range(50)])
    b = p.stage([spin(2_000, trace, 50 + i) for i in range(50)])
    p.after(a, b)
    h = p.upload(fab, first=a, iterations=5)
    fab.start()
    fab.publish(map=all_lane(fab, 0), caps=[0], order=[0], programs=[h])
    wait_idle(fab, 0)
    it = fab.progress()["lanes"][0]["iterations"]
    fab.stop()
    tr = decode_trace(trace)
    assert it == 5
    assert (tr["count"][:100] == 5).all()



def test_program_switch_at_boundary(fab):
    """A repeating program is replaced at an iteration boundary."""
    trace = trace_buffer(1024)
    p1 = Program()
    p1.stage([spin(5_000, trace, i) for i in range(64)])
    h1 = p1.upload(fab, first=0, iterations=0)
    p2 = Program()
    p2.stage([spin(5_000, trace, 512 + i) for i in range(64)])
    h2 = p2.upload(fab, first=0, iterations=3)
    fab.start()
    fab.publish(map=all_lane(fab, 0), caps=[0], order=[0], programs=[h1])
    time.sleep(0.05)
    fab.publish(map=all_lane(fab, 0), caps=[0], order=[0], programs=[h2])
    wait_until(lambda: fab.lane_stats(0)["running"] == 0, what="switch")
    gen = fab.progress()["lanes"][0]["gen"]
    fab.stop()
    tr = decode_trace(trace)
    assert gen == fab.generation(h2)
    assert (tr["count"][512:576] == 3).all()
    assert tr["count"][:64].min() >= 1


def test_dynamic_region_precedes_reset_tiles():
    """Device tiles (the expert expander, the decode attention plan) fill
    the reserved region at run time. The reset stage's own tiles must lie
    outside it, or an iterating program loses the re-arm of its stages."""
    p = Program()
    a = p.stage([Tile(KIND["nop"])] * 3)
    b = p.stage()
    p.after(a, b)
    p.reserve(5)
    dyn = p.dynamic_first
    tiles, stages, _, _, reset, _ = p.pack(a, iterations=4)
    st = stages.numpy()
    r_first, r_count = int(st[reset, 0]), int(st[reset, 1])
    assert dyn == 3 and r_count > 0
    assert r_first >= dyn + 5
    assert tiles.shape[0] == r_first + r_count
    kinds = tiles.numpy()[r_first:r_first + r_count, 0] & 0xFFFF
    assert (kinds == KIND["zero_runtime"]).all()
