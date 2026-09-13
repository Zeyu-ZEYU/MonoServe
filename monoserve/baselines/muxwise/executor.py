# SPDX-License-Identifier: Apache-2.0
"""vLLM's single-process executor running MuxWise's steps.

vLLM's worker loads the model (with the Local plugin's expert placement),
sizes and binds the KV cache, and builds the attention backends as usual;
after its warm-up, the executor sets up the stream groups and captures the
decode graphs, and every execute_model call runs one MuxWise step on the
worker's model. The server runs with --enforce-eager, so vLLM captures no
graphs of its own.
"""
from concurrent.futures import Future

from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.outputs import ModelRunnerOutput

from monoserve.baselines.muxwise.config import load


class MuxWiseExecutor(UniProcExecutor):
    supports_pp = False

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        from monoserve.baselines.muxwise.runtime import MuxRuntime
        conf = (self.vllm_config.additional_config or {}).get("muxwise", {})
        runner = self.driver_worker.worker.model_runner
        self.mux = MuxRuntime(runner, self.vllm_config, load(conf.get("config")),
                              capture=not conf.get("no_graphs", False))
        return result

    def execute_model(self, scheduler_output, non_block=False):
        step = getattr(scheduler_output, "muxwise", None)
        ids, toks = self.mux.step(step) if step is not None else ([], [])
        out = ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
                                sampled_token_ids=toks)
        if not non_block:
            return out
        f = Future()
        f.set_result(out)
        return f

    def sample_tokens(self, grammar_output, non_block=False):
        raise RuntimeError("MuxWise samples inside its step; run vLLM with --no-async-scheduling")

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block=False,
                       **extra):
        if method == "muxwise_profile":   # the offline profile (profile.py)
            from monoserve.baselines.muxwise.profile import measure
            result = [measure(self.mux, *args, **(kwargs or {}))]
            if not non_block:
                return result
            f = Future()
            f.set_result(result)
            return f
        return super().collective_rpc(method, timeout=timeout, args=args, kwargs=kwargs,
                                      non_block=non_block, **extra)
