"""Requests carry their SLO targets as vLLM extra args (CPU only)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import loadgen  # noqa: E402


def test_payload_sends_targets():
    req = {"prompt": "hello", "max_tokens": 4}
    assert "vllm_xargs" not in loadgen.payload(req, "m")
    body = loadgen.payload(req, "m", target=(1.5, 0.2))
    assert body["vllm_xargs"] == {"ttft_slo": 1.5, "tpot_slo": 0.2}
    assert body["max_tokens"] == 4 and body["stream"]
    # a request without a TPOT target (fewer than two output tokens in solo)
    assert loadgen.payload(req, "m", target=(1.5, None))["vllm_xargs"] == {"ttft_slo": 1.5}
    # extra args given through the sampling settings are kept
    merged = loadgen.payload(req, "m", {"temperature": 0.0, "vllm_xargs": {"a": 1}},
                             target=(1.0, 0.1))
    assert merged["vllm_xargs"] == {"a": 1, "ttft_slo": 1.0, "tpot_slo": 0.1}
