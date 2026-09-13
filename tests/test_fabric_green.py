"""The fabric launched once per green-context partition, as in the /KF
ablation: workers keep their indices across launches, each lane's tiles run
only on its own partition's workers and SMs, and the lanes run again after
a re-carve into other partition sizes."""
import time

import numpy as np
import pytest
import torch

from monoserve.fabric import KIND, Fabric, Program, Tile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

NO_LANE = 255
SPAN = 4096      # trace records per lane and round


def decode(t):
    t = t.cpu().numpy().astype(np.uint64)
    low = np.uint64(0xFFFFFFFF)
    return {"worker": (t[:, 2] & low).astype(np.int64),
            "sm": (t[:, 2] >> np.uint64(32)).astype(np.int64),
            "count": (t[:, 3] & low).astype(np.int64)}


def spin_program(fab, trace, first, n, ns=20_000):
    p = Program()
    p.stage([Tile(KIND["spin"], i=(first + i, 0), a=(ns, trace.data_ptr())) for i in range(n)])
    return p.upload(fab, first=0)


def wait_runs(fab, lane, runs, timeout=30.0):
    t0 = time.time()
    while fab.lane_stats(lane)["running"] or fab.progress()["lanes"][lane]["iterations"] < runs:
        if time.time() - t0 > timeout:
            raise TimeoutError(f"lane {lane} did not finish")
        time.sleep(0.01)


def test_lanes_on_green_partitions():
    from monoserve import _C
    fab = Fabric()
    parts = _C.GreenPartitions()
    rounds = ((16, 32), (40, 8))
    trace = torch.zeros(2 * len(rounds) * SPAN, 4, dtype=torch.int64, device="cuda")
    n = 2000
    try:
        for rnd, sizes in enumerate(rounds):
            granted = parts.create(list(sizes))
            assert len(granted) == 2 and min(granted) > 0
            first = [0, granted[0]]
            mapping = [NO_LANE] * fab.num_workers
            for lane in (0, 1):
                mapping[first[lane]:first[lane] + granted[lane]] = [lane] * granted[lane]
            base = [(2 * rnd + lane) * SPAN for lane in (0, 1)]
            progs = [spin_program(fab, trace, base[lane], n) for lane in (0, 1)]
            for lane in (0, 1):
                fab.start_on(parts.stream(lane), parts.context(lane), granted[lane], first[lane])
            # no borrowing: each lane's workers exist only in its partition
            fab.publish(map=mapping, caps=[0, 0], order=[NO_LANE], programs=progs)
            for lane in (0, 1):
                wait_runs(fab, lane, rnd + 1)
            t0 = time.time()
            fab.stop()        # a re-carve: every launch ends, the partitions go
            parts.destroy()
            assert time.time() - t0 < 5.0
            tr = decode(trace)
            sms = []
            for lane in (0, 1):
                s = slice(base[lane], base[lane] + n)
                assert (tr["count"][s] == 1).all(), (rnd, lane)
                w = tr["worker"][s]
                assert w.min() >= first[lane] and w.max() < first[lane] + granted[lane], (rnd, lane)
                sms.append(set(tr["sm"][s].tolist()))
            assert not sms[0] & sms[1], "the partitions share no SM"
    finally:
        fab.stop()
        parts.destroy()
