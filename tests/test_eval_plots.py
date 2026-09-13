"""Smoke tests of every evaluation figure script, the alpha table and the
metrics CLI, on synthetic result files the test writes in the harness's
layout and record format (CPU only)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.common import (run_path, solo_path, workload_path,  # noqa: E402
                         write_json, write_jsonl)
from eval.workloads import REFERENCE_ID  # noqa: E402

# method -> (meta.hot_share, rate at which its load reaches 1)
METHODS = {"MonoServe": (None, 8.0), "Baseline": (None, 4.0),
           "MonoServe/ML": (None, 3.0), "MonoServe-hot0": (0.0, 5.0),
           "MonoServe-hot40": (40.0, 7.0), "MonoServe-hot80": (80.0, 8.0)}
RATES = [1, 2, 4, 8, 16]


def record(i, req, solo, slow, done):
    n, send = 20, i * 0.1
    ttft, tpot = solo["ttft"] * slow, solo["tpot"] * slow
    chunk_t = [round(send + ttft + k * tpot, 6) for k in range(n)]
    return {"seq": i, "id": req["id"], "endpoint": i % 2,
            "outstanding": [0, 0], "prompt_tokens": req["prompt_tokens"],
            "max_tokens": req["max_tokens"], "arrival": send, "send": send,
            "first": chunk_t[0], "finish": chunk_t[-1] if done else None,
            "output_tokens": n, "finish_reason": "stop" if done else None,
            "usage": None, "error": None if done else "unfinished",
            "chunk_t": chunk_t, "chunk_n": [1] * n}


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval") / "results"
    rng = np.random.default_rng(0)
    for model in ("model-a", "model-b"):
        for ds in ("sharegpt", "longbench"):
            reqs = [{"id": f"{ds}-{i}", "prompt": "p", "prompt_tokens": int(n),
                     "max_tokens": 1000}
                    for i, n in enumerate(rng.integers(10, 5000, 40))]
            ref = {"id": REFERENCE_ID, "prompt": "p", "prompt_tokens": 1024,
                   "max_tokens": 16}
            write_jsonl(workload_path(root, model, ds),
                        {"dataset": ds, "reference": ref}, reqs)
            solo = {r["id"]: {"prompt_tokens": r["prompt_tokens"],
                              "ttft": 0.05 + r["prompt_tokens"] * 1e-4,
                              "tpot": 0.01} for r in reqs + [ref]}
            write_json(solo_path(root, model, ds), {"meta": {}, "solo": solo})
            for method, (share, cap) in METHODS.items():
                for rate in RATES:
                    load = rate / cap
                    recs = [record(i, r, solo[r["id"]],
                                   1 + 6 * load ** 2 * rng.random(),
                                   load < 1.5 or rng.random() > 0.2)
                            for i, r in enumerate(reqs)]
                    meta = {} if share is None else {"hot_share": share}
                    write_jsonl(run_path(root, model, ds, method, rate),
                                {"method": method, "model": model,
                                 "dataset": ds, "rate": rate, "meta": meta},
                                recs)
    return root


def run(*args):
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=600)


SCRIPTS = [("fig09_attainment", ["--methods", "Baseline", "MonoServe"]),
           ("fig10_p99_tails", []),
           ("fig11_ablation", ["--methods", "MonoServe", "MonoServe/ML"]),
           ("fig12_hot_tier", []),
           ("tab02_alpha", ["--methods", "MonoServe", "Baseline"])]


@pytest.mark.parametrize("script,extra", SCRIPTS)
def test_figure_scripts(results, tmp_path, script, extra):
    p = run(f"eval.plots.{script}", "--results", str(results), "--out-dir",
            str(tmp_path), "--labels", "model-a=Model A", *extra)
    assert p.returncode == 0, p.stderr
    files = list(tmp_path.iterdir())
    assert len(files) == 1 and files[0].stat().st_size > 0
    if script == "tab02_alpha":
        tex = files[0].read_text()
        assert tex.startswith("\\begin{tabular}") and "$\\alpha=10$" in tex
        assert "Model A, LongBench v2" in tex and "Baseline" in tex
        assert "ShareGPT" not in tex


def test_metrics_cli(results, tmp_path):
    out = tmp_path / "summary.json"
    p = run("eval.metrics", "--results", str(results), "--model", "model-a",
            "--dataset", "sharegpt", "--alpha", "5", "--json", str(out))
    assert p.returncode == 0, p.stderr
    assert "rate at 90% attainment" in p.stdout and out.exists()
