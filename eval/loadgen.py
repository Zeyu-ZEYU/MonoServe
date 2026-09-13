"""Open-loop load generator for OpenAI-compatible completion endpoints.

Requests arrive as a Poisson stream at a cluster-wide rate. A client-side
router sends each one to the endpoint with the fewest outstanding requests,
the policy of vLLM's data-parallel API server; ties rotate through the
endpoints. Every request streams /v1/completions with
stream_options.include_usage and never sets ignore_eos.

Each record holds, in seconds from the start of the run: arrival (the
scheduled arrival), send, first (the first chunk that carries tokens, its
text possibly empty), finish, and chunk_t / chunk_n, the time and the token
count of every such chunk. A chunk with a choice counts one token, unless
the server reports cumulative usage in every chunk (continuous_usage, a vLLM
extension), in which case it counts the increase. output_tokens is the
completion_tokens of the final usage chunk, or the sum of chunk_n when the
server sends no usage. A request still running drain_timeout seconds after
the last arrival is cancelled and recorded with error "unfinished"; any
other failure is recorded in error as well.
"""
import asyncio
import json
import time

import aiohttp
import numpy as np

from .common import log

DEFAULT_SAMPLING = {"temperature": 0.0}


def add_client_args(ap):
    ap.add_argument("--api-key")
    ap.add_argument("--served-model",
                    help="model name in the requests (default: the first "
                         "one the endpoint lists under /v1/models)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--extra-body", type=json.loads, default={},
                    help='JSON object merged into every request, e.g. '
                         '\'{"top_p": 0.8}\'')
    ap.add_argument("--continuous-usage", action="store_true",
                    help="ask for usage in every chunk (vLLM's "
                         "continuous_usage_stats) to count tokens per chunk "
                         "exactly")


def client_kwargs(args):
    return {"model": args.served_model,
            "sampling": {"temperature": args.temperature, **args.extra_body},
            "continuous_usage": args.continuous_usage,
            "api_key": args.api_key}


def arrival_times(rate, seed, num=None, duration=None):
    """Poisson arrival times, the first at 0, for num requests or for all
    arrivals before duration seconds. The gaps are unit exponentials drawn
    from the seed and divided by the rate, so the points of a sweep share
    one arrival pattern and differ only in its scale."""
    if (num is None) == (duration is None):
        raise ValueError("give num or duration")
    rng = np.random.default_rng(seed)
    if num is not None:
        u = rng.standard_exponential(max(0, num - 1))
    else:
        u = np.empty(0)
        while u.sum() <= duration * rate:
            u = np.concatenate([u, rng.standard_exponential(4096)])
    t = np.concatenate([[0.0], np.cumsum(u)]) / rate
    return t[:num] if duration is None else t[t < duration]


class Router:
    """Least outstanding requests; ties go to the first endpoint after the
    last one picked."""

    def __init__(self, n):
        self.out = [0] * n
        self.last = -1

    def pick(self):
        n = len(self.out)
        best = None
        for k in range(1, n + 1):
            i = (self.last + k) % n
            if best is None or self.out[i] < self.out[best]:
                best = i
        self.out[best] += 1
        self.last = best
        return best

    def done(self, i):
        self.out[i] -= 1


def base_url(endpoint):
    url = (endpoint if "://" in endpoint else f"http://{endpoint}").rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def client_session(api_key=None):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    # No connection cap and no overall timeout: a request may wait in a
    # server queue for minutes before its first token.
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0),
        timeout=aiohttp.ClientTimeout(total=None, sock_connect=60),
        headers=headers)


async def served_model(session, endpoint):
    async with session.get(base_url(endpoint) + "/v1/models") as r:
        r.raise_for_status()
        return (await r.json())["data"][0]["id"]


def payload(req, model, sampling=None, continuous_usage=False, target=None):
    """target: (TTFT, TPOT) targets in seconds, sent as vLLM extra args
    (servers that plan for SLOs read them; the others ignore them)."""
    opts = {"include_usage": True}
    if continuous_usage:
        opts["continuous_usage_stats"] = True
    body = {"model": model, "prompt": req["prompt"],
            "max_tokens": req["max_tokens"], "stream": True,
            "stream_options": opts,
            **(DEFAULT_SAMPLING if sampling is None else sampling)}
    if target is not None:
        xargs = {k: v for k, v in zip(("ttft_slo", "tpot_slo"), target) if v is not None}
        body["vllm_xargs"] = {**body.get("vllm_xargs", {}), **xargs}
    return body


def new_record(req, seq=0, endpoint=0, outstanding=None, arrival=0.0):
    return {"seq": seq, "id": req["id"], "endpoint": endpoint,
            "outstanding": outstanding, "prompt_tokens": req["prompt_tokens"],
            "max_tokens": req["max_tokens"], "arrival": arrival, "send": None,
            "first": None, "finish": None, "output_tokens": None,
            "finish_reason": None, "usage": None, "error": None,
            "chunk_t": [], "chunk_n": []}


async def stream(session, url, body, t0, rec):
    """Send one streaming completion and fill rec in place."""
    chunk_t, chunk_n = rec["chunk_t"], rec["chunk_n"]
    seen = 0          # cumulative completion tokens reported so far
    rec["send"] = round(time.perf_counter() - t0, 6)
    try:
        async with session.post(url, json=body) as resp:
            if resp.status != 200:
                rec["error"] = f"HTTP {resp.status}: {(await resp.text())[:300]}"
                return
            async for line in resp.content:
                t = time.perf_counter() - t0
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    rec["finish"] = round(t, 6)
                    break
                msg = json.loads(data)
                if "error" in msg:
                    rec["error"] = f"server: {str(msg['error'])[:300]}"
                    return
                usage = msg.get("usage")
                if usage:
                    rec["usage"] = usage
                choices = msg.get("choices")
                if not choices:
                    continue
                if choices[0].get("finish_reason"):
                    rec["finish_reason"] = choices[0]["finish_reason"]
                n = 1
                if usage and usage.get("completion_tokens") is not None:
                    n, seen = usage["completion_tokens"] - seen, usage["completion_tokens"]
                if n > 0:
                    chunk_t.append(round(t, 6))
                    chunk_n.append(n)
            if rec["finish"] is None:
                if rec["finish_reason"] is None:
                    rec["error"] = "stream closed before the end"
                else:     # the server ended the stream without [DONE]
                    rec["finish"] = round(time.perf_counter() - t0, 6)
    except asyncio.CancelledError:
        rec["error"] = "unfinished"
        raise
    except Exception as e:  # connection failures and malformed streams
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        if rec["error"] is not None:
            rec["finish"] = None
        rec["first"] = chunk_t[0] if chunk_t else None
        usage = rec["usage"] or {}
        rec["output_tokens"] = usage.get("completion_tokens", sum(chunk_n))


async def _report(records, t0, every):
    while True:
        await asyncio.sleep(every)
        done = sum(r["finish"] is not None or r["error"] is not None
                   for r in records)
        log(f"{time.perf_counter() - t0:7.0f}s: sent {len(records)}, "
            f"done {done}")


async def run(requests, endpoints, rate, *, num=None, duration=None, seed=0,
              model=None, sampling=None, continuous_usage=False,
              drain_timeout=600.0, api_key=None, progress=30.0, targets=None):
    """Drive one sweep point. Requests are taken in workload order (cycling
    if the run needs more); targets maps request ids to their (TTFT, TPOT)
    targets, sent with the requests. Returns (records, info) where info
    holds the served model name and the wall time of the run."""
    times = arrival_times(rate, seed, num=num, duration=duration)
    urls = [base_url(e) + "/v1/completions" for e in endpoints]
    router = Router(len(endpoints))
    records, tasks = [], []

    async def one(rec, body):
        try:
            await stream(session, urls[rec["endpoint"]], body, t0, rec)
        finally:
            router.done(rec["endpoint"])

    async with client_session(api_key) as session:
        if model is None:
            model = await served_model(session, endpoints[0])
        t0 = time.perf_counter()
        reporter = (asyncio.create_task(_report(records, t0, progress))
                    if progress else None)
        for seq, at in enumerate(times):
            delay = t0 + at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            req = requests[seq % len(requests)]
            outstanding = list(router.out)
            rec = new_record(req, seq, router.pick(), outstanding,
                             round(float(at), 6))
            records.append(rec)
            body = payload(req, model, sampling, continuous_usage,
                           targets.get(req["id"]) if targets else None)
            tasks.append(asyncio.create_task(one(rec, body)))
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=drain_timeout)
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        if reporter:
            reporter.cancel()
        wall = time.perf_counter() - t0
    return records, {"served_model": model, "wall_s": round(wall, 3)}
