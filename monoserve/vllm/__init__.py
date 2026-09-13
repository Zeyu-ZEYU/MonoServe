"""MonoServe inside vLLM.

vLLM keeps its model configuration, tokenizer, request handling, and
OpenAI-compatible API server. The control plane replaces its scheduler
(scheduler.MonoServeScheduler) and the fabric replaces its model execution
(executor.MonoServeExecutor): both run in vLLM's engine-core process and
share one engine, which the executor builds and registers here.

    python -m monoserve.serve --model /path/to/model --calibration calibration.json
"""
import logging

# the engine's messages (memory plan, loading, decisions) through vLLM's
# log handlers, which are configured for vLLM's own loggers only
_log = logging.getLogger("monoserve")
if not _log.handlers:
    for _h in logging.getLogger("vllm").handlers:
        _log.addHandler(_h)
    if _log.handlers:
        _log.setLevel(logging.INFO)
        _log.propagate = False

_engine = None
_finished = {}   # request id -> finish reason the engine decided (rejected, length)


def set_engine(engine):
    global _engine
    _engine = engine


def engine():
    if _engine is None:
        raise RuntimeError("the MonoServe executor has not built its engine; run vLLM with "
                           "--distributed-executor-backend monoserve.vllm.executor.MonoServeExecutor")
    return _engine


def engine_finished():
    return _finished
