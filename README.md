# MonoServe

MonoServe serves mixture-of-experts (MoE) models on a GPU whose expert set
lives in CPU DRAM. HBM holds the non-expert weights, the KV cache, and a
hot tier of experts; the kernels read every other expert in place over the
CPU-GPU link (NVLink-C2C on a GH200, PCIe on other systems). Prefill and
decode batches run concurrently as lanes of one persistent kernel, the
kernel fabric (MonoFab), which divides the SMs among the lanes at tile
boundaries and gates each lane's reads of CPU-resident experts with link
credits. A contention-aware control plane (admission, a time estimator,
and a plan search) decides at every admission event which batches run, on
how many SMs, with how many link credits, and with which kernel form.
MonoServe runs behind vLLM's OpenAI-compatible API server.

## Requirements

- An NVIDIA Hopper GPU. The kernels are built for `sm_90a` and read pinned
  CPU memory, so the CPU-GPU link sets how fast CPU-resident experts stream.
- Linux, a CUDA toolkit (12.8 or newer), and a driver with green contexts
  (CUDA 12.5 or newer), which the /KF ablation, the MuxWise baseline, and
  the Section 2 co-runs use.
- Python 3.10 or newer, PyTorch, and vLLM 0.29.0.
- Host memory for the pinned expert set: 58 GB for
  Qwen3-30B-A3B-Instruct-2507 in BF16 and 100 GB for GLM-4.5-Air in FP8.

## Installation

From the root of this repository (a clone or an unpacked download):

```bash
pip install vllm==0.29.0            # brings a matching PyTorch
bash scripts/fetch_deps.sh          # CUTLASS and ThunderKittens at pinned commits
pip install --no-build-isolation -e .
```

`MAX_JOBS` bounds the parallel compile jobs. The package also registers a
vLLM general plugin for the Local baseline; it acts only when a server's
configuration asks for it.

## Tests

```bash
python -m pytest -q tests
```

The fabric, runtime, checkpoint-loading, engine, calibration, and MuxWise
tests need a Hopper GPU; the control-plane, MuxWise-configuration, and
evaluation tests run on the CPU. `tests/test_vllm_serve.py` starts vLLM's
API server on a small random model and runs only when
`MONOSERVE_TOKENIZER_DIR` points at a Qwen3-MoE checkpoint directory, whose
configuration and tokenizer it borrows.

## Serving a model

Supported checkpoints are Qwen3-MoE (BF16) and GLM-4.5 MoE (BF16, or FP8 in
the compressed-tensors format with one scale per output channel), for
example `Qwen/Qwen3-30B-A3B-Instruct-2507` and `zai-org/GLM-4.5-Air-FP8`.

1. Calibrate the time estimator once per GPU type and model shape. The
   runs use random weights of the model's shape.

   ```bash
   python -m monoserve.calibrate --model /path/to/model --out calibration.json
   ```

2. Optionally, measure the expert activation profiles that choose the hot
   tier and order the staging copies, by serving sample prompts (one JSON
   object per line, `{"prompt": ...}`). Without a profile, both treat the
   experts as equally likely.

   ```bash
   python -m monoserve.profile --model /path/to/model --calibration calibration.json \
       --prompts prompts.jsonl --out profile.json
   ```

3. Serve:

   ```bash
   python -m monoserve.serve --model /path/to/model --calibration calibration.json \
       --profile profile.json --port 8000 [-- extra vllm serve arguments]
   ```

   The hot tier takes the HBM left after the non-expert weights, the
   staging buffers, the lanes' workspaces, and the KV cache (`--kv-gb`);
   `--hot-fraction` or `--hbm-budget-gb` shrink it. The HBM that
   `--hot-fraction` frees goes to the KV cache unless `--kv-gb` is given. Loading stacks every
   layer's experts into pinned memory and takes several minutes for the
   models above.

   Requests go to the OpenAI-compatible endpoints. A request may carry its
   SLO targets, in seconds, as `vllm_xargs`; without them, its targets are
   `--alpha` times its estimated solo latency. Admission rejects the
   requests whose targets no plan can meet, and vLLM reports them with the
   finish reason `abort`. Sampling uses the temperature only.

   ```bash
   curl http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{
       "model": "/path/to/model", "prompt": "The capital of France is", "max_tokens": 16,
       "temperature": 0, "vllm_xargs": {"ttft_slo": 2.0, "tpot_slo": 0.1}}'
   ```

## Baselines and ablations

| Method | Command |
|---|---|
| Local: stock vLLM with the expert set in CPU DRAM and the same hot tier | `python -m monoserve.baselines.serve_local --model M --profile P --hot-fraction F` |
| MuxWise: prefill-decode multiplexing on green contexts, ported onto Local | `python -m monoserve.baselines.muxwise.profile --model M --profile P --hot-fraction F --out divisions.yaml`, then `python -m monoserve.baselines.serve_muxwise --model M --divisions divisions.yaml --profile P --hot-fraction F` |
| EP: stock vLLM with the experts sharded across the GPUs of a node | `python -m monoserve.baselines.serve_ep --model M --gpus 8` |
| /ML: one mixed batch at a time on every SM | `python -m monoserve.serve ... --engine ml` |
| /MCA: planning without gamma, and no credit gate | `python -m monoserve.serve ... --no-contention-awareness` |
| /KF: the lanes on green-context partitions instead of the fabric | `python -m monoserve.serve ... --engine kf` |
| One kernel form for CPU-resident experts | `python -m monoserve.serve ... --forms 64` (a tile height) or `--forms 0` (asymmetric) |
| No staging | `python -m monoserve.serve ... --no-staging` |

## Evaluation and Section 2 measurements

[`eval/README.md`](eval/README.md) describes the evaluation harness: the
request sets from ShareGPT and LongBench v2, solo latency, rate sweeps
against any set of OpenAI-compatible servers, and the metrics, figures, and
table of Section 5, computed from the recorded runs.
[`motivation/README.md`](motivation/README.md) describes the measurements of
Section 2 and Appendix A; `motivation/data/gh200/` holds the points the
paper plots.

## Code map

| Paper | Code |
|---|---|
| §3.1 admission | `csrc/control/admission.{h,cpp}`, driven by `monoserve/engine.py` |
| §3.1 expert placement | `monoserve/placement.py`, `monoserve/deploy.py`, the host loop in `csrc/host/host_loop.cpp` |
| §3.1 time estimator (Eqs. 1 and 2) | `csrc/control/estimator.{h,cpp}`, calibrated by `monoserve/calibrate/` |
| §3.1 plan search (Algorithm 1, Appendix B) | `csrc/control/search.{h,cpp}` |
| §3.2 kernel fabric: lanes, stage rings, link slots, epochs | `csrc/fabric/runtime.cuh`, `csrc/host/fabric.{h,cpp}` |
| §3.2 tiles: symmetric and asymmetric expert GEMMs, attention, the rest of a layer | `csrc/fabric/tiles/` |
| A transformer as lane programs of tiles | `monoserve/runtime/lane.py`, `monoserve/fabric/` |
| §4 integration with vLLM | `monoserve/vllm/`, `monoserve/serve.py` |
| §5 baselines and ablations | `monoserve/baselines/`, `monoserve/ablations.py` |

## License

Apache License 2.0; see [`LICENSE`](LICENSE).
