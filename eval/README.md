# Evaluation harness

The client side of the evaluation. It drives any set of OpenAI-compatible
servers, one per GPU server, that stream `/v1/completions`; MonoServe and
the vLLM-based baselines all expose this endpoint. It builds the request
sets, measures every request's solo latency, sweeps the request rate,
records per-request latency, and computes the metrics, figures and table of
the evaluation from the recorded results only.

## Requirements

- Python 3.10 or newer with `numpy`, `matplotlib` and `aiohttp`. The client
  needs no GPU and neither PyTorch nor the `monoserve` package.
- `transformers` (the tokenizer of the served model) and `huggingface_hub`
  (the datasets) for building the request sets.
- Servers reachable over HTTP that serve the same model under the same name.
  The client runs in one Python process; give it a CPU core of its own, so
  that it sends on schedule and timestamps chunks as they arrive. Each run
  records every request's send lag behind its scheduled arrival, and
  `eval.metrics` prints the largest one.

## Results layout

All scripts run from the repository root and share one directory
(`--results`, default `results/`):

```
results/<model>/<dataset>/workload.jsonl           eval.workloads
results/<model>/<dataset>/solo.json                eval.solo
results/<model>/<dataset>/<method>/rate_<r>.jsonl  eval.run_sweep
```

`<model>` is a name you choose per served model and `<dataset>` is
`sharegpt` or `longbench`. `<method>` is the method name with characters
other than letters, digits and `._+-` replaced by `_`. Method names are free
strings; the run headers keep them as given, and the figures take them from
there.

A run file starts with a header line (method, model, dataset, rate, number
of requests, seed, alpha, endpoints, served model name, sampling settings,
start time, wall time, `meta`, and the workload settings), followed by one
record per request: `id`, `endpoint`, `outstanding` (every endpoint's
outstanding requests when the router chose), `arrival`, `send`, `first`,
`finish` (seconds from the start of the run), `chunk_t` and `chunk_n` (time
and token count of every chunk with tokens), `output_tokens` (from the usage
the server reports), `finish_reason` and `error`.

## Running

1. Build the request set of each model and dataset. Prompts are shuffled
   with `--seed`. Prompts longer than `--max-prompt-tokens` are dropped, and
   every request asks for `--max-model-len` minus its prompt length output
   tokens, so outputs run to the end of the sequence.

   ```bash
   python -m eval.workloads --dataset sharegpt --tokenizer Qwen/Qwen3-30B-A3B-Instruct-2507 \
       --max-model-len 131072 --num 2000 --model qwen3-30b
   python -m eval.workloads --dataset longbench --tokenizer Qwen/Qwen3-30B-A3B-Instruct-2507 \
       --max-model-len 131072 --max-prompt-tokens 128000 --num 0 --model qwen3-30b
   ```

   ShareGPT prompts are first human turns and LongBench v2 prompts use the
   zero-shot multiple-choice template of the LongBench v2 evaluation. They
   are sent as raw text; `--chat-template` wraps each one in the tokenizer's
   chat template instead.

2. Measure solo latency on one server of the configuration that sets the
   targets, with nothing else running on it. Requests go one at a time.
   The results are cached per request, so an interrupted run resumes.

   ```bash
   python -m eval.solo --model qwen3-30b --dataset sharegpt --endpoint http://<server-1>:8000
   ```

   When one solo run per request costs too much, `--per-bucket K` measures
   only the first K requests of each prompt-length bucket (powers of two
   from 1K to 256K tokens). The metrics and figures then need
   `--solo-mode buckets`, which interpolates every request's solo latency
   over prompt length through the bucket medians.

3. Sweep the request rate of each method against its servers. Rates count
   the whole cluster.

   ```bash
   python -m eval.run_sweep --model qwen3-30b --dataset sharegpt --method MonoServe \
       --endpoints http://<server-1>:8000 http://<server-2>:8000 \
       --rates 4 8 16 32 48 64 --num-requests 2000
   ```

   `--duration S` replaces `--num-requests` with every arrival within S
   seconds. Requests still running `--drain-timeout` seconds after the last
   arrival are cancelled and count as not finished. Existing rate files are
   skipped unless `--overwrite` is given. `--served-model`, `--api-key`,
   `--temperature` (default 0), `--extra-body` and `--continuous-usage`
   (per-chunk usage, a vLLM extension that counts the tokens of multi-token
   chunks exactly) apply to both `eval.run_sweep` and `eval.solo`. For the
   hot-tier sensitivity, run one sweep per tier size under its own method
   name and pass the tier's share of the expert set in percent, for example
   `--method MonoServe-hot41 --meta hot_share=41.2`.

4. Print the metrics of every method of a model and dataset:

   ```bash
   python -m eval.metrics --model qwen3-30b --dataset sharegpt --alpha 5
   ```

5. Draw the figures and the table:

   ```bash
   python -m eval.plots.fig09_attainment --out-dir figures --methods EP Local MuxWise MonoServe
   python -m eval.plots.fig10_p99_tails --out-dir figures --methods EP Local MuxWise MonoServe
   python -m eval.plots.fig11_ablation --out-dir figures \
       --methods MonoServe MonoServe/ML MonoServe/MCA MonoServe/KF
   python -m eval.plots.fig12_hot_tier --out-dir figures
   python -m eval.plots.tab02_alpha --out-dir figures --methods MonoServe MuxWise
   ```

   Every script takes `--results`, `--models` (rows, default all),
   `--labels model=Label`, `--datasets`, `--methods` (default all, in
   sorted order), `--alpha`, `--solo-mode` and `--name`.

| Paper item | Script | Content |
|---|---|---|
| Fig. 9 | `fig09_attainment.py` | SLO attainment against cluster rate, rows = models, columns = datasets; dotted line at 90% |
| Fig. 10 | `fig10_p99_tails.py` | P99 TTFT over solo TTFT (top rows) and P99 TPOT (bottom rows) against rate; dotted target lines |
| Fig. 11 | `fig11_ablation.py` | Fig. 9 for the ablation variants |
| Fig. 12 | `fig12_hot_tier.py` | Rate at 90% attainment against the hot tier's share, relative to the rightmost point |
| Table 2 | `tab02_alpha.py` | Rate at 90% attainment against alpha (LaTeX tabular, LongBench by default) |
| Figs. 16, 17 | `fig10_p99_tails.py --methods <ablation variants>` | Tails behind Fig. 11 |

## Metrics

- TTFT is the time from sending a request to its first chunk with tokens.
  TPOT is (t_last - t_first) / (n_out - 1) over the chunks with tokens,
  with n_out the output tokens the server reports. A request with fewer
  than two output tokens has no TPOT.
- A request meets its SLO when its TTFT is within alpha times the larger of
  its solo TTFT and the solo TTFT of a 1024-token reference prompt, and its
  TPOT is within alpha times its solo TPOT. The reference makes prompts
  under 1K tokens take the 1K prompt's TTFT target. The default alpha is 5.
  `eval.run_sweep` sends every request's targets with it, and MonoServe's
  admission plans for them, so its runs are made for one alpha; the
  baselines ignore the targets, and their runs can be scored with any
  alpha. For the alpha table, run one MonoServe sweep per alpha
  (`--alpha`), or send no targets (`--no-send-targets`), in which case
  MonoServe plans for `--alpha` times its own solo estimates.
- SLO attainment is the share of all sent requests that meet their SLO.
  Requests that did not finish count as violations.
- P99 values use the nearest rank over all requests; a request that did not
  finish counts as infinite, and the figures draw an infinite P99 at the
  top edge of the panel. TTFT is reported over the solo TTFT that sets the
  request's target, TPOT in milliseconds.
- Goodput is the rate sustained at 90% attainment: the first sweep point
  below 90% and the point before it are joined linearly. It is undefined
  when the lowest rate is already below 90%, and only a lower bound when no
  swept rate falls below 90%.

## Tests

```bash
python -m pytest -q tests/test_eval_*.py
```

They run on the CPU against stub streaming servers and synthetic result
files.
