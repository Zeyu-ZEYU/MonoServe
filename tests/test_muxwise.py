"""MuxWise on vLLM, end to end on a small random model: prompts arrive
together, a prefill batch runs layer slice by layer slice beside decode
steps on a green-context division, and every request's greedy tokens equal
those of stock vLLM with the same model and kernels (the Local baseline)."""
import json

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def generate(model, extra, prompts, **kw):
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(model), skip_tokenizer_init=True, enforce_eager=True,
              enable_prefix_caching=False, max_model_len=1024, gpu_memory_utilization=0.3,
              cpu_offload_gb=100000, cpu_offload_params={"w13_weight", "w2_weight"},
              additional_config=extra, **kw)
    sp = SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True, detokenize=False)
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp)
    toks = [list(o.outputs[0].token_ids) for o in outs]
    llm.llm_engine.engine_core.shutdown()
    del llm
    return toks


def test_muxwise_matches_vllm(tmp_path):
    pytest.importorskip("vllm")
    from tiny_model import make_tiny_qwen3_moe
    model = make_tiny_qwen3_moe(None, tmp_path / "tiny", layers=4)
    S = torch.cuda.get_device_properties(0).multi_processor_count
    p = (S - 40) // 8 * 8
    divisions = tmp_path / "divisions.json"
    # one division, and a small slice budget: every prefill batch takes
    # several steps, each beside a decode step
    divisions.write_text(json.dumps({"sm_group_num": 3, "manual_divisions": [[p, S - p, 1]],
                                     "split_forward_token_budget": 256,
                                     "max_prefill_tokens": 200}))
    g = torch.Generator().manual_seed(0)
    prompts = [torch.randint(10, 1000, (n,), generator=g).tolist() for n in (5, 37, 100, 260, 90)]
    local = {"local_moe": {"hot_fraction": 0.5}}
    ref = generate(model, local, prompts)
    mux = generate(model, {**local, "muxwise": {"config": str(divisions)}}, prompts,
                   distributed_executor_backend="monoserve.baselines.muxwise.executor.MuxWiseExecutor",
                   scheduler_cls="monoserve.baselines.muxwise.scheduler.MuxWiseScheduler",
                   async_scheduling=False)
    assert mux == ref
