"""Shared helpers for the Section 2 measurements.

Logging, GPU-spin launch decoupling, trimmed statistics, per-module
CUDA-event timers, and a single green-context SM partition.
"""
import statistics
import time

import torch

# torch.cuda._sleep spins for a number of SM clock cycles. A lower actual
# clock only lengthens a spin, which is the safe direction for the
# launch-decoupling spins used below.
SM_CLOCK_HZ = 1.98e9


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sleep_cycles(sec):
    return int(sec * SM_CLOCK_HZ)


def burn_in(sec=20):
    """Bring the GPU clocks and power state up before measuring."""
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    t0 = time.time()
    while time.time() - t0 < sec:
        a = a @ a * 1e-3
    torch.cuda.synchronize()
    del a
    torch.cuda.empty_cache()
    log(f"burn-in {sec}s done")


def summarize(vals):
    return {
        "mean": statistics.mean(vals),
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals),
        "raw": vals,
    }


def trimmed_summarize(vals):
    """Drop one maximum and one minimum when there are at least four
    samples and average the rest into "value", the reported number."""
    s = sorted(vals)
    core = s[1:-1] if len(s) >= 4 else s
    out = summarize(vals)
    out["value"] = statistics.mean(core)
    return out


def kept_spread(vals):
    """Relative spread of the samples that trimmed_summarize keeps."""
    s = sorted(vals)
    core = s[1:-1] if len(s) >= 4 else s
    return (core[-1] - core[0]) / (sum(core) / len(core))


def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0).item()


class ModuleTimer:
    """CUDA-event pairs recorded by forward hooks around every attention
    and MoE sublayer. collect() returns the summed span per module kind
    in milliseconds; call it after torch.cuda.synchronize()."""

    KINDS = ("attn", "moe")

    def __init__(self, model, max_forwards=160):
        layers = model.model.layers
        n = max_forwards * len(layers)
        self.pools = {
            k: [(torch.cuda.Event(enable_timing=True),
                 torch.cuda.Event(enable_timing=True)) for _ in range(n)]
            for k in self.KINDS
        }
        self.idx = {k: 0 for k in self.KINDS}
        self.handles = []
        for layer in layers:
            self._attach(layer.self_attn, "attn")
            self._attach(layer.mlp, "moe")

    def _attach(self, module, kind):
        def pre(mod, args):
            self.pools[kind][self.idx[kind]][0].record()

        def post(mod, args, output):
            self.pools[kind][self.idx[kind]][1].record()
            self.idx[kind] += 1

        self.handles.append(module.register_forward_pre_hook(pre))
        self.handles.append(module.register_forward_hook(post))

    def reset(self):
        self.idx = {k: 0 for k in self.KINDS}

    def collect(self):
        out = {}
        for k in self.KINDS:
            out[k] = sum(a.elapsed_time(b)
                         for a, b in self.pools[k][:self.idx[k]])
        return out

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class NullTimer:
    def reset(self):
        pass


class GreenCtx:
    """Confine every kernel launched inside the context manager to
    num_sms SMs through a CUDA green context. A full-device request is a
    no-op."""

    def __init__(self, num_sms, total_sms):
        self.active = num_sms < total_sms
        self.gc = None
        if self.active:
            self.gc = torch.cuda.GreenContext.create(num_sms=num_sms,
                                                     device_id=0)

    def __enter__(self):
        if self.active:
            self.gc.set_context()
        return self

    def __exit__(self, *exc):
        if self.active:
            self.gc.pop_context()
        return False
