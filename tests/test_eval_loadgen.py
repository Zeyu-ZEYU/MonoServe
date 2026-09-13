"""End-to-end tests of the load generator and of solo profiling against stub
OpenAI-compatible streaming servers (aiohttp, CPU only). A stub waits a
first-token delay, then streams one or several tokens per chunk with a
per-token delay that grows with the number of streams it is serving; the
first word of a prompt is the number of tokens to generate."""
import asyncio
import contextlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import loadgen, metrics, solo  # noqa: E402
from eval.workloads import REFERENCE_ID  # noqa: E402


class Stub:
    def __init__(self, ttft=0.01, tpot=0.002, slowdown=0.0, per_chunk=1):
        self.ttft, self.tpot = ttft, tpot
        self.slowdown, self.per_chunk = slowdown, per_chunk
        self.active = self.peak = self.served = 0

    def factor(self):
        return 1 + self.slowdown * (self.active - 1)

    async def models(self, request):
        return web.json_response({"object": "list",
                                  "data": [{"id": "stub", "object": "model"}]})

    async def completions(self, request):
        body = await request.json()
        if body["max_tokens"] < 1:
            return web.json_response(
                {"error": {"message": "max_tokens must be positive"}},
                status=400)
        n = min(int(body["prompt"].split()[0]), body["max_tokens"])
        opts = body.get("stream_options") or {}
        resp = web.StreamResponse(
            headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.served += 1
        try:
            await asyncio.sleep(self.ttft * self.factor())
            sent = 0
            while sent < n:
                k = min(self.per_chunk, n - sent)
                sent += k
                chunk = {"object": "text_completion", "usage": None,
                         "choices": [{"index": 0, "text": " tok" * k,
                                      "finish_reason":
                                          "stop" if sent == n else None}]}
                if opts.get("continuous_usage_stats"):
                    chunk["usage"] = {"prompt_tokens": 2,
                                      "completion_tokens": sent}
                await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
                if sent < n:
                    await asyncio.sleep(self.tpot * k * self.factor())
            if opts.get("include_usage"):
                usage = {"prompt_tokens": 2, "completion_tokens": n,
                         "total_tokens": 2 + n}
                last = {"choices": [], "usage": usage}
                await resp.write(f"data: {json.dumps(last)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
        finally:
            self.active -= 1
        return resp


@contextlib.asynccontextmanager
async def serving(*stubs):
    runners = []
    try:
        for s in stubs:
            app = web.Application()
            app.router.add_post("/v1/completions", s.completions)
            app.router.add_get("/v1/models", s.models)
            r = web.AppRunner(app, access_log=None, handler_cancellation=True)
            await r.setup()
            await web.TCPSite(r, "127.0.0.1", 0).start()
            runners.append(r)
        yield [f"http://127.0.0.1:{r.addresses[0][1]}" for r in runners]
    finally:
        for r in runners:
            await r.cleanup()


def requests(ns, max_tokens=64):
    return [{"id": f"q{i}", "prompt": f"{n} words", "prompt_tokens": 2,
             "max_tokens": max_tokens} for i, n in enumerate(ns)]


def drive(stubs, reqs, rate, **kw):
    async def go():
        async with serving(*stubs) as urls:
            return await loadgen.run(reqs, urls, rate, progress=0, **kw)
    return asyncio.run(go())


def test_arrival_times_are_poisson():
    t = loadgen.arrival_times(50.0, seed=3, num=20000)
    assert t[0] == 0 and len(t) == 20000
    g = np.diff(t)
    assert g.mean() == pytest.approx(1 / 50, rel=0.03)
    assert g.std() / g.mean() == pytest.approx(1.0, rel=0.05)
    # Kolmogorov-Smirnov distance to the exponential CDF, 1% critical value
    s, n = np.sort(g), len(g)
    cdf = 1 - np.exp(-50 * s)
    ks = max(np.max(np.arange(1, n + 1) / n - cdf),
             np.max(cdf - np.arange(n) / n))
    assert ks < 1.63 / math.sqrt(n)
    # one pattern per seed, scaled by the rate
    assert np.allclose(loadgen.arrival_times(100.0, seed=3, num=20000), t / 2)
    d = loadgen.arrival_times(20.0, seed=4, duration=100.0)
    assert d.max() < 100 and abs(len(d) - 2000) < 5 * math.sqrt(2000)
    assert np.allclose(loadgen.arrival_times(20.0, seed=4, num=len(d)), d)


def test_router_prefers_the_least_loaded_and_rotates_ties():
    r = loadgen.Router(3)
    assert [r.pick() for _ in range(3)] == [0, 1, 2]
    r.done(1)
    assert r.pick() == 1
    r.done(0)
    r.done(2)
    assert r.pick() == 2 and r.pick() == 0


def test_stream_records_timestamps_and_usage():
    recs, info = drive([Stub(ttft=0.05, tpot=0.01)], requests([20]), 1.0,
                       num=1)
    r = recs[0]
    assert info["served_model"] == "stub"
    assert r["error"] is None and r["finish_reason"] == "stop"
    assert r["output_tokens"] == 20 and r["chunk_n"] == [1] * 20
    assert r["usage"]["completion_tokens"] == 20
    assert r["send"] <= r["first"] <= r["chunk_t"][-1] <= r["finish"]
    assert metrics.ttft(r) == pytest.approx(0.05, abs=0.03)
    assert metrics.tpot(r) == pytest.approx(0.01, rel=0.3)
    gaps = np.diff(r["chunk_t"])
    assert np.all(gaps > 0) and np.median(gaps) == pytest.approx(0.01, rel=0.3)


def test_chunks_with_several_tokens():
    plain = drive([Stub(per_chunk=4)], requests([10]), 1.0, num=1)[0][0]
    exact = drive([Stub(per_chunk=4)], requests([10]), 1.0, num=1,
                  continuous_usage=True)[0][0]
    # without per-chunk usage a chunk counts one token; usage fixes the total
    assert plain["chunk_n"] == [1, 1, 1] and plain["output_tokens"] == 10
    assert exact["chunk_n"] == [4, 4, 2] and exact["output_tokens"] == 10
    assert metrics.tpot(exact) == pytest.approx(
        (exact["chunk_t"][-1] - exact["chunk_t"][0]) / 9)


def test_poisson_arrivals_end_to_end():
    recs, _ = drive([Stub(ttft=0.001, tpot=0.001)], requests([2] * 50),
                    100.0, num=300, seed=7)
    assert len(recs) == 300 and all(r["error"] is None for r in recs)
    sched = loadgen.arrival_times(100.0, 7, num=300)
    assert [r["arrival"] for r in recs] == pytest.approx(list(sched), abs=1e-6)
    assert max(r["send"] - r["arrival"] for r in recs) < 0.1
    assert np.diff([r["send"] for r in recs]).mean() == pytest.approx(
        0.01, rel=0.1)
    assert recs[50]["id"] == "q0"          # the workload cycles


def test_least_outstanding_routing():
    fast, slow = Stub(tpot=0.001), Stub(tpot=0.02)
    recs, _ = drive([fast, slow], requests([20] * 10), 200.0, num=200, seed=1)
    for r in recs:
        assert r["outstanding"][r["endpoint"]] == min(r["outstanding"])
    n_fast = sum(r["endpoint"] == 0 for r in recs)
    assert (n_fast, 200 - n_fast) == (fast.served, slow.served)
    assert n_fast > 1.5 * (200 - n_fast) > 0


def test_unfinished_and_failed_requests():
    recs, _ = drive([Stub(ttft=0.01, tpot=1.0)], requests([100] * 3), 100.0,
                    num=3, drain_timeout=0.3)
    assert all(r["error"] == "unfinished" and r["finish"] is None
               and r["first"] is not None for r in recs)
    t = {r["id"]: solo.Target(0.01, 1.0, 1.0) for r in recs}
    s = metrics.summarize(recs, t)
    assert s["attainment"] == 0 and s["finished"] == 0
    assert s["p99_ttft_ratio"] == math.inf == s["p99_tpot_ms"]
    bad, _ = drive([Stub()], requests([5], max_tokens=0), 1.0, num=1)
    assert bad[0]["error"].startswith("HTTP 400") and bad[0]["finish"] is None


def test_solo_profile_and_attainment_under_load(tmp_path):
    work = requests([5 + i % 10 for i in range(24)])
    ref = {"id": REFERENCE_ID, "prompt": "3 words", "prompt_tokens": 1024,
           "max_tokens": 16}
    path = tmp_path / "solo.json"
    stub = Stub(ttft=0.02, tpot=0.004, slowdown=0.5)

    async def go():
        async with serving(stub) as urls:
            await solo.profile([ref] + work, urls[0], path, warmup=1)
            await solo.profile([ref] + work, urls[0], path)   # all cached
            n = stub.served
            low, _ = await loadgen.run(work, urls, 5.0, num=12, seed=2,
                                       progress=0)
            high, _ = await loadgen.run(work, urls, 1000.0, num=24, seed=2,
                                        progress=0)
        return n, low, high

    served, low, high = asyncio.run(go())
    assert served == 1 + 25                  # one warm-up, then 25 once each
    cache = solo.load_solo(path)
    assert set(cache["solo"]) == {REFERENCE_ID} | {r["id"] for r in work}
    e = cache["solo"]["q3"]
    assert e["output_tokens"] == 8 and e["ttft"] == pytest.approx(0.02, abs=0.02)
    assert e["tpot"] == pytest.approx(0.004, rel=0.5)
    tg = solo.targets(work, cache, alpha=2)
    lo, hi = metrics.summarize(low, tg), metrics.summarize(high, tg)
    assert lo["attainment"] >= 0.9 and hi["attainment"] < 0.5
    assert hi["p99_tpot_ms"] > 2 * lo["p99_tpot_ms"]
    assert stub.peak >= 10
