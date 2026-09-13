"""Lane programs: DAGs of stages of tiles, packed for upload to the fabric.

A stage is a set of tiles that become ready together; a stage becomes
ready when all of its predecessor stages have completed. The builder adds
the program's reset stage, which depends on every sink stage, re-arms the
bookkeeping of all stages, and ends one iteration. A program with
iterations=0 repeats until the host publishes a newer program for its lane;
a finished program can be published again.

Tiles are given either as Tile objects or, for large stages, as uint64
arrays of shape (n, 8) built with tile_array(); both pack to the 64-byte
TileDesc of csrc/fabric/types.h. A program may reserve a dynamic region of
tiles that device tiles (the expert expander, the decode attention plan)
fill at run time; its first index is dynamic_first after all stages exist.
"""
from dataclasses import dataclass, field

import numpy as np
import torch

KIND = {"nop": 0, "spin": 1, "read": 2, "probe": 3, "zero_runtime": 4}
STAGE_LINK, STAGE_RESET, STAGE_MARK = 1, 2, 4
_U32 = np.uint64(0xFFFFFFFF)


@dataclass
class Tile:
    kind: int
    i: tuple = (0, 0, 0, 0)
    a: tuple = (0, 0, 0, 0, 0)
    flags: int = 0


def tile_array(kind, i0=0, i1=0, i2=0, i3=0, a0=0, a1=0, a2=0, a3=0, a4=0):
    """Pack broadcastable fields into an (n, 8) uint64 array of tiles."""
    cols = np.broadcast_arrays(*[np.asarray(v, dtype=np.int64)
                                 for v in (i0, i1, i2, i3, a0, a1, a2, a3, a4)])
    n = cols[0].size
    out = np.zeros((n, 8), dtype=np.uint64)
    c = [np.ascontiguousarray(x).reshape(-1).astype(np.uint64) for x in cols]
    out[:, 0] = np.uint64(kind & 0xFFFF)
    out[:, 1] = (c[0] & _U32) | ((c[1] & _U32) << np.uint64(32))
    out[:, 2] = (c[2] & _U32) | ((c[3] & _U32) << np.uint64(32))
    for k in range(5):
        out[:, 3 + k] = c[4 + k]
    return out


def _tiles_to_array(tiles):
    if isinstance(tiles, np.ndarray):
        return tiles
    if not tiles:
        return np.zeros((0, 8), dtype=np.uint64)
    rows = []
    for t in tiles:
        i = (list(t.i) + [0, 0, 0, 0])[:4]
        a = (list(t.a) + [0] * 5)[:5]
        r = tile_array(t.kind, *i, *a)
        r[:, 0] |= np.uint64((t.flags & 0xFF) << 24)
        rows.append(r)
    return np.concatenate(rows)


@dataclass
class _Stage:
    tiles: np.ndarray
    flags: int = 0
    mark: int = 0
    succ: list = field(default_factory=list)
    preds: int = 0
    label: object = None


class Program:
    def __init__(self):
        self.stages = []
        self.reserved = 0

    def stage(self, tiles=(), link=False, mark=None, label=None):
        """Add a stage; returns its id. link=True marks DRAM-expert tiles;
        label (for example (name, layer)) names it in stage traces."""
        flags = (STAGE_LINK if link else 0) | (STAGE_MARK if mark is not None else 0)
        self.stages.append(_Stage(_tiles_to_array(tiles), flags, mark or 0, label=label))
        return len(self.stages) - 1

    def labels(self):
        """Stage labels by stage id, the reset stage last (see trace.py)."""
        return [s.label for s in self.stages] + [("reset", -1)]

    def after(self, pred, succ):
        """succ becomes ready only after pred has completed."""
        self.stages[pred].succ.append(succ)
        self.stages[succ].preds += 1

    def reserve(self, n):
        """Reserve n tiles for run-time generation (see dynamic_first)."""
        self.reserved += n

    @property
    def dynamic_first(self):
        return sum(len(s.tiles) for s in self.stages)

    def pack(self, first, iterations=1, reset_tiles=8):
        """Return (tiles int64 [n, 8], stages int32 [m, 8], succ int32 [k],
        first_stage, reset_stage, iterations)."""
        for sid, s in enumerate(self.stages):
            if sid != first and s.preds == 0:
                raise ValueError(f"stage {sid} has no predecessor and is not the "
                                 f"first stage, so it would never become ready")
        n_user = len(self.stages)
        reset = n_user
        sinks = [i for i, s in enumerate(self.stages) if not s.succ]
        # Every iteration, the last one included, ends by re-arming the
        # bookkeeping of all stages, so a finished program can run again
        # (on the same lane or another one).
        zero = []
        n_all = n_user + 1
        chunk = -(-n_all // max(1, reset_tiles))
        for b in range(0, n_all, chunk):
            zero.append(Tile(KIND["zero_runtime"], i=(b, min(n_all, b + chunk), reset, 0)))
        stages = self.stages + [_Stage(_tiles_to_array(zero), STAGE_RESET, 0, [], len(sinks))]
        succ_lists = [list(s.succ) for s in stages]
        for s in sinks:
            succ_lists[s].append(reset)

        blocks, rows, succ = [], [], []
        first_tile = 0
        for sid, s in enumerate(stages):
            if sid == reset and self.reserved:
                # The dynamic region follows the user stages' tiles, at
                # dynamic_first, and precedes the reset stage's own tiles.
                blocks.append(np.zeros((self.reserved, 8), dtype=np.uint64))
                first_tile += self.reserved
            t = s.tiles.copy()
            if len(t):
                t[:, 0] = (t[:, 0] & _U32) | (np.uint64(sid) << np.uint64(32))
            blocks.append(t)
            rows.append([first_tile, len(t), len(succ), len(succ_lists[sid]), s.preds,
                         s.flags, s.mark, 0])
            succ.extend(succ_lists[sid])
            first_tile += len(t)
        packed = np.concatenate(blocks) if blocks else np.zeros((0, 8), dtype=np.uint64)
        tiles_t = torch.from_numpy(packed.view(np.int64).copy())
        stages_t = torch.tensor(rows, dtype=torch.int64).to(torch.int32)
        succ_t = torch.tensor(succ if succ else [0], dtype=torch.int32)
        return tiles_t, stages_t, succ_t, first, reset, iterations

    def upload(self, fabric, first, iterations=1, reset_tiles=8):
        tiles, stages, succ, f, r, it = self.pack(first, iterations, reset_tiles)
        return fabric.upload(tiles, stages, succ, f, r, it)
