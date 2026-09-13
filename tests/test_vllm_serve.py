"""vLLM's OpenAI-compatible API server with MonoServe's scheduler and
executor, on a small random checkpoint: completions come back from the
fabric, with vLLM's stop checks and usage accounting.

Needs vLLM and a GPU, and MONOSERVE_TOKENIZER_DIR pointing at a Qwen3-MoE
checkpoint directory whose config and tokenizer the small model borrows."""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
    pytest.mark.skipif("MONOSERVE_TOKENIZER_DIR" not in os.environ,
                       reason="set MONOSERVE_TOKENIZER_DIR to a Qwen3-MoE checkpoint"),
]


def free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def post(url, body, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def test_openai_completions(tmp_path):
    pytest.importorskip("vllm")
    from test_engine import calibration_for
    from tiny_model import make_tiny_qwen3_moe
    model = make_tiny_qwen3_moe(os.environ["MONOSERVE_TOKENIZER_DIR"], tmp_path / "tiny")
    S = torch.cuda.get_device_properties(0).multi_processor_count
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps(calibration_for(S)))
    port = free_port()
    log_path = tmp_path / "server.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "monoserve.serve", "--model", str(model), "--calibration",
             str(cal), "--kv-gb", "0.2", "--max-model-len", "1024", "--token-budget", "512",
             "--host", "127.0.0.1", "--port", str(port), "--", "--max-num-seqs", "16"],
            stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    try:
        t0 = time.time()
        while True:
            assert proc.poll() is None, log_path.read_text()[-4000:]
            try:
                urllib.request.urlopen(url + "/health", timeout=2)
                break
            except OSError:
                time.sleep(1)
            assert time.time() - t0 < 900, log_path.read_text()[-4000:]
        # SLO targets ride in vllm_xargs; without them the engine targets alpha
        # times its solo estimate, which the placeholder calibration makes
        # far tighter than the small model runs, and admission rejects
        def body(p):
            return {"model": str(model), "prompt": p, "max_tokens": 8, "temperature": 0.0,
                    "vllm_xargs": {"ttft_slo": 5.0, "tpot_slo": 1.0}}

        prompts = ["Hello", "The capital of France is", "1 2 3 4 5", "Once upon a time"]
        with ThreadPoolExecutor(len(prompts)) as pool:
            outs = list(pool.map(lambda p: post(url + "/v1/completions", body(p)), prompts))
        for out in outs:
            assert out["usage"]["completion_tokens"] == 8, out
            assert out["choices"][0]["finish_reason"] == "length", out
        # the same prompt twice under greedy sampling gives the same text
        again = post(url + "/v1/completions", body(prompts[1]))
        assert again["choices"][0]["text"] == outs[1]["choices"][0]["text"]
    finally:
        proc.terminate()
        try:
            proc.wait(120)
        except subprocess.TimeoutExpired:
            proc.kill()
