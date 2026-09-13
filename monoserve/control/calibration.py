"""Calibration files of the time estimator.

A calibration is measured inside the fabric (monoserve.calibrate: the solo
sweeps, one credit sweep per kernel form, and the gamma probe) and stored as
JSON in these units: SM counts, microseconds, MTQ entries (32-byte
sectors), TFLOP/s, and GB/s.

    sms, R_H_us, R_C_us, q_max, beta_gbps, reference_form,
    layer_overhead_us, tail_overhead_us
    gamma:  {exposure_entry_us: [...], stretch: [...]}
    dense, attn_prefill, attn_decode, expert[form]:
            {sms: [...], tflops: [...], gbps: [...], q_per_sm}
    rho[form]: {credits: [...], gbps: [...]}
    q_form[form]: entries one DRAM-expert tile keeps in flight

Forms are "16", "32", "64", "128" (symmetric templates by tile height) and
"asym" (the asymmetric kernel).
"""
import json

FORM_KEYS = {"16": 16, "32": 32, "64": 64, "128": 128, "asym": 0}


def _rates(ctl, d):
    k = ctl.KernelRates()
    sms = [float(s) for s in d["sms"]]
    k.flops = ctl.Curve(sms, [v * 1e12 for v in d["tflops"]], True)
    k.bytes = ctl.Curve(sms, [v * 1e9 for v in d["gbps"]], True)
    k.q_per_sm = float(d["q_per_sm"])
    return k


def load_calibration(src):
    """A Calibration from a JSON file path or an already parsed dict."""
    from monoserve import _C
    ctl = _C.control
    if isinstance(src, dict):
        d = src
    else:
        with open(src) as f:
            d = json.load(f)
    c = ctl.Calibration()
    c.sms = int(d["sms"])
    c.R_H = float(d["R_H_us"]) * 1e-6
    c.R_C = float(d["R_C_us"]) * 1e-6
    c.gamma = ctl.Curve([x * 1e-6 for x in d["gamma"]["exposure_entry_us"]],
                        [float(y) for y in d["gamma"]["stretch"]], False)
    c.q_max = float(d["q_max"])
    c.dense = _rates(ctl, d["dense"])
    c.attn_prefill = _rates(ctl, d["attn_prefill"])
    c.attn_decode = _rates(ctl, d["attn_decode"])
    c.expert = {FORM_KEYS[k]: _rates(ctl, v) for k, v in d["expert"].items()}
    c.rho = {FORM_KEYS[k]: ctl.Curve([float(x) for x in v["credits"]],
                                     [g * 1e9 for g in v["gbps"]], True)
             for k, v in d["rho"].items()}
    c.q_form = {FORM_KEYS[k]: float(v) for k, v in d["q_form"].items()}
    c.beta = float(d["beta_gbps"]) * 1e9
    c.reference_form = int(d.get("reference_form", 64))
    c.layer_overhead = float(d.get("layer_overhead_us", 0.0)) * 1e-6
    c.tail_overhead = float(d.get("tail_overhead_us", 0.0)) * 1e-6
    return c


def save_calibration(data, path):
    """Write a calibration dict (the format above) as JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True)
