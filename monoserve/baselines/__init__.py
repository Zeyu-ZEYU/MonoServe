"""The baselines of the evaluation, built on vLLM.

  Local    stock vLLM with the expert set in CPU DRAM, which vLLM's
           offloading reads through unified virtual addressing (UVA),
           and MonoServe's hot tier (local_moe.py, serve_local.py)
  MuxWise  prefill-decode multiplexing on green contexts, ported from its
           SGLang implementation onto Local's model and expert placement
           (muxwise/, serve_muxwise.py)
  EP       stock vLLM sharding the experts across GPUs with expert
           parallelism (serve_ep.py)

The Local plugin is registered with vLLM as a general plugin (see setup.py)
and acts only when vLLM's additional config asks for it.
"""


def register():
    """vLLM general-plugin entry point."""
    from monoserve.baselines import local_moe
    local_moe.register()
