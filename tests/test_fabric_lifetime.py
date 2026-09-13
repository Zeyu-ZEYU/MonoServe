"""Fabric lifetimes: a fabric goes as soon as its last reference does, and
freeing a stopped fabric never waits for another fabric's kernel (cudaFree
waits for the device to go idle, which a running persistent kernel never
does)."""
import gc
import pathlib
import subprocess
import sys
import textwrap
import weakref

import pytest
import torch

from monoserve.fabric import Fabric
from monoserve.runtime.config import MoEConfig
from monoserve.runtime.kv import KVCache
from monoserve.runtime.lane import Lane
from monoserve.runtime.requests import RequestTable
from monoserve.runtime.weights import ModelWeights

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

CFG = MoEConfig(hidden=512, intermediate=256, num_experts=16, top_k=4, num_layers=2,
                num_heads=4, num_kv_heads=2, vocab=1024, rope_theta=1e6)
ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_lanes_hold_no_cycle():
    def build():
        fab = Fabric(pool_bytes=256 << 20)
        w = ModelWeights.synthetic(fab, CFG, [list(range(8))] * CFG.num_layers)
        kv = KVCache(fab, CFG, 64)
        req = RequestTable(fab, 8, 16)
        Lane(fab, CFG, w, kv, req, 0, "prefill", max_tokens=256, max_reqs=4)
        Lane(fab, CFG, w, kv, req, 1, "decode", max_reqs=8, max_len=1024)
        return weakref.ref(fab)

    gc.disable()
    try:
        ref = build()
        assert ref() is None, [type(o).__name__ for o in gc.get_referrers(ref())]
    finally:
        gc.enable()


def test_freeing_a_stopped_fabric_while_another_runs():
    code = textwrap.dedent("""
        from monoserve.fabric import Fabric
        f1 = Fabric(pool_bytes=64 << 20)
        f1.start()
        f1.stop()
        f2 = Fabric(pool_bytes=64 << 20)
        f2.start()
        del f1
        f2.stop()
        print("ok")
    """)
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-2000:]
