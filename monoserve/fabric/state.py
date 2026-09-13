"""Snapshots of fabric state for tests and tools: a program's stage table
and bookkeeping, its tiles, and the entries waiting in a lane's rings.
All reads go through the fabric's control stream, so they are safe while
the kernel runs (the values may be mid-update)."""
import numpy as np

STAGE_BITS = 20
TAG_MASK = 0xFFF


def program_state(fab, handle):
    """Per-stage arrays of an uploaded program: first, count, succ_first,
    succ_count, deps_init, flags, mark (StageStatic) and tag, next, done,
    deps (StageRuntime)."""
    info = fab.program_info(handle)
    m = info["n_stages"]
    st = np.frombuffer(fab.read(info["stages"], 32 * m), dtype=np.uint32).reshape(m, 8)
    rt = np.frombuffer(fab.read(info["runtime"], 16 * m), dtype=np.uint32).reshape(m, 4)
    names = ("first", "count", "succ_first", "succ_count", "deps_init", "flags", "mark")
    out = {n: st[:, i].copy() for i, n in enumerate(names)}
    out.update(next=rt[:, 0].copy(), tag=rt[:, 1] & TAG_MASK, done=rt[:, 2].copy(),
               deps=rt[:, 3].copy())
    return out


def program_tiles(fab, handle, first=0, count=None):
    """Tiles [first, first + count) of a program as a uint64 [n, 8] array."""
    info = fab.program_info(handle)
    n = info["n_tiles"] - first if count is None else count
    raw = fab.read(info["tiles"] + 64 * first, 64 * n)
    return np.frombuffer(raw, dtype=np.uint64).reshape(n, 8)


def ring_entries(fab, lane, link=False):
    """(head, tail, [(stage, tag), ...]) of a lane's stage ring."""
    v = fab.lane_ring(lane, link)
    e = np.asarray(v["entries"], dtype=np.uint32)
    stages = (e & ((1 << STAGE_BITS) - 1)).tolist()
    tags = (e >> STAGE_BITS).tolist()
    return v["head"], v["tail"], list(zip(stages, tags))
