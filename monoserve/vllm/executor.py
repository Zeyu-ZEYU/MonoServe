"""A vLLM executor that runs the model on the fabric.

It builds the engine in vLLM's engine-core process, reports the engine's
KV cache (pages of 64 tokens) as the cache vLLM's scheduler accounts for,
and turns every execute_model call into one round of the engine's loop,
returning the tokens the lanes produced since the last round. No vLLM GPU
worker, model runner, or CUDA graph exists.
"""
import logging
import time
from concurrent.futures import Future

from vllm.v1.executor.abstract import Executor
from vllm.v1.outputs import ModelRunnerOutput

from monoserve import vllm as registry

log = logging.getLogger(__name__)

# Longest wait for new tokens in one round: bounds how long an arrival waits
# for the engine-core loop to pick it up.
ROUND_WAIT_S = 0.0005


def _local_path(vllm_config):
    """The checkpoint directory, fetched by vLLM's model loader when the
    model is a Hugging Face id."""
    from monoserve.runtime.checkpoint import resolve
    return resolve(vllm_config.model_config.model, vllm_config.model_config.revision,
                   vllm_config.load_config.download_dir)


class MonoServeExecutor(Executor):
    supported_tasks = ("generate",)

    def _init_executor(self):
        import torch
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        from monoserve.deploy import DeployOptions, build_engine
        opts = DeployOptions.from_dict(self.vllm_config.additional_config.get("monoserve", {}),
                                       max_len=self.model_config.max_model_len)
        self.engine = build_engine(_local_path(self.vllm_config), opts, check_stops=False,
                                   log=lambda *a: log.info(" ".join(str(x) for x in a)))
        registry.set_engine(self.engine)
        cfg = self.engine.cfg
        self.spec = FullAttentionSpec(block_size=64, num_kv_heads=cfg.num_kv_heads,
                                      head_size=cfg.head_dim, dtype=torch.bfloat16)
        self.layers = [f"model.layers.{i}.self_attn.attn" for i in range(cfg.num_layers)]

    # KV cache: the engine owns its pages; vLLM only sizes its accounting.
    def get_kv_cache_specs(self):
        return [{name: self.spec for name in self.layers}]

    def get_supported_kv_cache_layouts(self):
        return [["LBNHC"]]   # per layer: [pages][64 tokens][KV heads][head dim]

    def set_kv_cache_layout(self, layout_name):
        pass

    def determine_available_memory(self):
        return [self.engine.conf.kv_pages * len(self.layers) * self.spec.page_size_bytes]

    def initialize_from_config(self, kv_cache_configs):
        if kv_cache_configs[0].num_blocks > self.engine.conf.kv_pages:
            raise RuntimeError("vLLM sized the KV cache larger than the engine's")

    def compile_or_warm_up_model(self):
        self.engine.start()

    def supports_draft_weight_updates(self):
        return False

    def reset_mm_cache(self):
        pass

    def reset_encoder_cache(self):
        pass

    def profile(self, is_start=True, profile_prefix=None):
        pass

    # ------------------------------------------------------------------
    def execute_model(self, scheduler_output, non_block=False):
        out = self.engine.step(wait=ROUND_WAIT_S if scheduler_output.total_num_scheduled_tokens
                               else 0.0)
        finished = registry.engine_finished()
        ids, tokens = [], []
        for rid, toks, reason in out:
            if reason in ("rejected", "length"):
                finished[rid] = reason
            ids.append(rid)
            tokens.append(list(toks))
        result = ModelRunnerOutput(req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
                                   sampled_token_ids=tokens)
        if not non_block:
            return result
        f = Future()
        f.set_result(result)
        return f

    def sample_tokens(self, grammar_output, non_block=False):
        raise RuntimeError("MonoServe samples on the fabric; run vLLM with --no-async-scheduling")

    def take_draft_token_ids(self):
        return None

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block=False):
        # The fabric occupies the GPU for good: a device-wide synchronize
        # would never return, and there are no workers to call.
        if method == "synchronize_device":
            result = None
        elif isinstance(method, str) and callable(getattr(self, method, None)):
            result = getattr(self, method)(*args, **(kwargs or {}))
        else:
            raise NotImplementedError(f"MonoServe executor has no {method}")
        if not non_block:
            return [result]
        f = Future()
        f.set_result([result])
        return f

    def check_health(self):
        if not self.engine.fab.running:
            raise RuntimeError("the fabric is not running")

    def shutdown(self):
        if getattr(self, "engine", None) is not None and self.engine.fab.running:
            self.engine.stop()
            time.sleep(0)
