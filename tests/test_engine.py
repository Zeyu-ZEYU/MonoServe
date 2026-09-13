"""The engine end to end on a small model: requests arrive, admission opens
prefill lanes (one prompt is long enough to run in two chunks), the host
loop stages experts, the requests join the decode batch, and every token
matches greedy decoding of a torch reference. The /ML and /KF ablation
engines run the same requests."""
import time

import pytest
import torch

from test_fabric_model import CFG, reference_logits

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def calibration_for(S):
    """A calibration file for a GPU of S SMs with rates of plausible shape.
    The test checks function, not timing, so measured values are not needed."""
    sms = sorted({s for s in (8, 16, 32, 64) if s < S} | {S})

    def rates(tflops, gbps, knee, q):
        return {"sms": sms, "tflops": [tflops * s / S for s in sms],
                "gbps": [gbps * min(1.0, s / knee) for s in sms], "q_per_sm": q}

    credits = [128, 256, 512, 1024, 2048, 4096, 8192]
    rho = {"credits": credits, "gbps": [min(50.0, n * 32 / 0.9e-6 / 1e9) for n in credits]}
    return {"sms": S, "R_H_us": 0.4, "R_C_us": 0.9, "q_max": 96,
            "gamma": {"exposure_entry_us": [2000, 8000, 16000], "stretch": [1.0, 1.4, 2.0]},
            "dense": rates(600, 3600, 70, 64), "attn_prefill": rates(450, 3000, 80, 48),
            "attn_decode": rates(150, 3200, 64, 80),
            "expert": {k: rates(300, 3400, 64, 96) for k in ("16", "32", "64", "128", "asym")},
            "rho": {k: rho for k in ("16", "32", "64", "128", "asym")},
            "q_form": {"16": 96, "32": 96, "64": 128, "128": 160, "asym": 64},
            "beta_gbps": 50.0, "reference_form": 64, "layer_overhead_us": 20.0,
            "tail_overhead_us": 30.0}


@pytest.mark.parametrize("kind", ["full", "ml", "kf"])
def test_engine_greedy(kind):
    from monoserve.ablations import GreenEngine, SingleLaneEngine
    from monoserve.engine import Engine, EngineConfig
    from monoserve.runtime.weights import ModelWeights
    S = torch.cuda.get_device_properties(0).multi_processor_count
    hot = [list(range(8)), list(range(4, 12))]
    conf = EngineConfig(max_slots=8, kv_pages=64, max_len=512, token_budget=256,
                        max_batch_requests=4, max_decode=8, staging_experts=2, pool_bytes=512 << 20)
    cls = {"full": Engine, "ml": SingleLaneEngine, "kf": GreenEngine}[kind]
    eng = cls(CFG, lambda fab, **kw: ModelWeights.synthetic(fab, CFG, hot, **kw),
              calibration_for(S), conf, log=lambda *a: None)
    g = torch.Generator().manual_seed(5)
    prompts = {i: torch.randint(0, CFG.vocab, (n,), generator=g).tolist()
               for i, n in enumerate([30, 90, 300])}
    steps = 6
    outs, done = {i: [] for i in prompts}, {}
    eng.start()
    try:
        for i, p in prompts.items():
            assert eng.add_request(i, p, max_tokens=steps, ttft=5.0, tpot=1.0) == []
        t0 = time.time()
        while len(done) < len(prompts):
            assert time.time() - t0 < 120, f"requests did not finish: {outs}"
            for rid, toks, reason in eng.step(wait=0.01):
                outs[rid] += toks
                if reason:
                    done[rid] = reason
        while not eng.idle():   # the decode lane lets go of the last requests
            assert time.time() - t0 < 120, "the engine did not release its requests"
            assert eng.step(wait=0.01) == []
        stats = eng.host_loop.stats() if getattr(eng, "host_loop", None) else None
    finally:
        eng.stop()
    assert all(r == "length" for r in done.values()), (done, outs)
    if kind != "ml":   # one lane on every SM stages nothing
        assert stats["copies"] > 0
    if kind == "kf":
        assert eng.stats["recarves"] > 1
    assert not eng.reqs and len(eng.free_slots) == conf.max_slots
    for i, p in prompts.items():
        assert len(outs[i]) == steps
        logits = reference_logits(eng.weights, CFG, p + outs[i][:-1])
        for j, t in enumerate(outs[i]):
            lg = logits[len(p) - 1 + j]
            best = int(lg.argmax())
            assert t == best or float(lg[best] - lg[t]) < 0.02 * float(lg.max() - lg.min()), (i, j)
