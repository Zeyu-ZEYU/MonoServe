"""Stage traces of the fabric, for profiling lane programs.

    trace = StageTrace(capacity=1 << 20)     # device buffer, allocated before the fabric starts
    trace.start(fab)                         # record from now on
    ...                                      # run programs
    records = trace.stop(fab)                # int64 [n, 5]: stage, lane, event, tag, device ns
    print(summary(records, lane.program_labels(handle), lane=0))

The kernel records a stage's activation (its last predecessor completed)
and its completion (its last tile completed). A stage's time is the span
between the two, which includes waiting for workers; stages that run side
by side overlap, so the spans of a label do not add up to wall time.
summary() reports, per stage label, the number of stages and their spans,
the span of every layer's experts (first w13 activation to last w2
completion), and each iteration's wall time.
"""
from collections import defaultdict

import numpy as np
import torch

ACTIVATE, COMPLETE = 0, 1


class StageTrace:
    def __init__(self, capacity=1 << 20, device="cuda"):
        self.capacity = capacity
        self.buf = torch.zeros(capacity, 2, dtype=torch.int64, device=device)

    def start(self, fab):
        fab.set_trace(self.buf.data_ptr(), self.capacity)

    def stop(self, fab):
        n = min(fab.trace_count(), self.capacity)
        fab.set_trace(0, 0)
        raw = np.frombuffer(fab.read(self.buf.data_ptr(), n * 16), dtype=np.uint64).reshape(n, 2)
        w = raw[:, 0]
        cols = [w & np.uint64(0xFFFFFFFF), (w >> np.uint64(32)) & np.uint64(0xFF),
                (w >> np.uint64(40)) & np.uint64(0xFF), (w >> np.uint64(48)) & np.uint64(0xFFF),
                raw[:, 1]]
        out = np.stack(cols, axis=1).astype(np.int64)
        return out[np.argsort(out[:, 4], kind="stable")]


def spans(records, labels, lane=None):
    """[(label, layer, tag, activation ns, completion ns)] of the stages
    that both activated and completed in the trace."""
    r = records if lane is None else records[records[:, 1] == lane]
    act, out = {}, []
    for s, _, ev, tag, t in r:
        key = (int(s), int(tag))
        if ev == ACTIVATE:
            act[key] = int(t)
        elif key in act:
            lab = labels[s] if s < len(labels) and labels[s] is not None else (f"stage{s}", -1)
            name, layer = lab if isinstance(lab, tuple) else (lab, -1)
            out.append((name, layer, int(tag), act.pop(key), int(t)))
    return out


def summary(records, labels, lane=None, experts=("w13", "red", "w2")):
    """A table of stage spans per label, expert spans per layer, and
    iteration wall times."""
    sp = spans(records, labels, lane)
    if not sp:
        return "no complete stages in the trace"
    per = defaultdict(list)
    for name, _, _, a, c in sp:
        per[name].append(c - a)
    lines = [f"{'stage':14s} {'n':>6s} {'mean us':>9s} {'p50 us':>9s} {'max us':>9s} {'sum ms':>9s}"]
    for name, d in sorted(per.items(), key=lambda kv: -sum(kv[1])):
        d = np.asarray(d) / 1e3
        lines.append(f"{name:14s} {len(d):6d} {d.mean():9.1f} {np.median(d):9.1f} {d.max():9.1f} "
                     f"{d.sum() / 1e3:9.3f}")
    ex = defaultdict(lambda: [np.inf, -np.inf])
    for name, layer, tag, a, c in sp:
        if name in experts:
            e = ex[(tag, layer)]
            e[0], e[1] = min(e[0], a), max(e[1], c)
    if ex:
        d = np.asarray([c - a for a, c in ex.values()]) / 1e3
        lines.append(f"experts of a layer: mean {d.mean():.1f} us, max {d.max():.1f} us "
                     f"({len(d)} layer iterations)")
    it = defaultdict(lambda: [np.inf, -np.inf])
    for _, _, tag, a, c in sp:
        e = it[tag]
        e[0], e[1] = min(e[0], a), max(e[1], c)
    d = np.asarray(sorted(c - a for a, c in it.values())) / 1e6
    lines.append("iterations (ms): " + ", ".join(f"{x:.2f}" for x in d[:8]))
    return "\n".join(lines)
