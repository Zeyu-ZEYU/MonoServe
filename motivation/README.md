# Section 2 measurements

These scripts measure how attention and MoE computations respond to their
SM share, how the two MoE kernels for CPU-resident experts compare, and how
co-running computations interfere through the L2 cache's miss-tracking
queue. Every figure of Section 2 and Appendix A has a script that produces
its data and a script that draws it.

| Figure | What is measured | Produced by |
|---|---|---|
| Fig. 2 | Attention and MoE latency versus SM share; one prefill pass and 128 decode steps at batch size 1, input lengths 1K to 128K; experts in CPU DRAM read by the symmetric kernel | `sm_share.py --moe-kernel sym` |
| Fig. 4 | Prefill MoE latency versus SM share, symmetric versus asymmetric kernel | `sm_share.py --moe-kernel sym` and `sm_share.py --moe-kernel asym --skip-decode` |
| Fig. 5 | Pair-wise co-location of prefill computations at 4K, each side on its own 48% of the SMs | `corun.py --suite pairs` |
| Fig. 6 | Attention slowdown beside a synthetic expert reader on 8 to 68 SMs, weights in CPU DRAM or in HBM | `corun.py --suite reader` |
| Fig. 13 | Batched decode (batch sizes 8 to 128) versus SM share | `sm_share_batch.py --batches 8 16 32 64 128 --moe-kernel sym` |
| Fig. 14 | Decode MoE latency, symmetric versus asymmetric kernel, batch sizes 1 to 256 | `sm_share_batch.py --sym-block-m auto` and `sm_share_batch.py --moe-kernel asym` |
| Fig. 15 | Attention slowdown beside a copy-engine transfer walking 1 to 128 GB of CPU DRAM | `corun.py --suite copy-engine` |
| Section 2.3 | Round trip of a miss to HBM and to CPU DRAM | `latency_probe.py` |
| Section 3.2 | Cost of re-carving green-context partitions | `green_ctx_recarve.py` |

## Files

- `moe_kernels.py`: the symmetric (output-stationary) and asymmetric
  (weight-stationary, K-split) Triton MoE kernels of Section 2.2. Running the
  file checks both kernels against vLLM's fused MoE kernel, with the weights
  in HBM and in pinned CPU DRAM.
- `pinned.py`: pinned host memory mapped into the GPU address space, which
  is how the kernels read CPU-resident experts in place.
- `model_harness.py`: Qwen3-MoE through Hugging Face transformers, with
  vLLM's FlashAttention-3 on a pre-allocated KV cache and fused MoE blocks.
- `partitions.py`: disjoint green-context SM partitions carved through the
  CUDA driver API. Running the file checks that two partitions report
  disjoint SM ids.
- `common.py`: timers, statistics, and a single green-context partition.
- `plots/`: `extract.py` turns result files into CSV tables, and the
  `fig*.py` scripts draw the figures from such tables.
- `data/gh200/`: the points plotted in the paper, measured on one GH200
  with Qwen3-30B-A3B-Instruct-2507.

## Requirements

- A Hopper GPU whose kernels can read pinned host memory. On a GH200 the
  reads cross NVLink-C2C; on a PCIe-attached GPU they cross PCIe, which has
  a lower bandwidth and a longer miss latency, so the numbers differ.
- The Python environment of the top-level README (PyTorch, vLLM, Triton,
  transformers) and a CUDA toolkit for the small inline extensions.
- The model weights, `Qwen/Qwen3-30B-A3B-Instruct-2507` (a Hugging Face id or
  a local path).
- Host memory for the pinned copies: 58 GB of experts for the sweeps, 19 GB
  for the reader suite, and 128 GB for the largest copy-engine footprint
  (reduce it with `--footprints-gb`).

## Drawing the figures from the shipped data

```bash
cd motivation
for f in plots/fig*.py; do python $f --out-dir figures; done
```

## Running the measurements

Everything at once, then tables and figures from the new results:

```bash
bash motivation/run_all.sh Qwen/Qwen3-30B-A3B-Instruct-2507 motivation/results
```

A single measurement, for example Fig. 2:

```bash
cd motivation
python sm_share.py --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --moe-kernel sym --out results/sm_share_sym.jsonl
python plots/extract.py --results results --out results/csv
python plots/fig02_sm_share.py --data results/csv --out-dir results/figures
```

Each sweep first validates the substituted kernels against the stock model
(`--phase validate`), then measures (`--phase sweep`); `--phase all` does
both. Sweeps resume when their output file already holds some points. Grids
are set with `--lengths`, `--batches`, and `--sms`; the defaults are the
grids of the paper. On a machine with several CPU sockets, bind each run to
the socket attached to the GPU so the pinned copies stay in its memory, for
example `numactl --cpunodebind=0 --membind=0 python sm_share.py ...`; on a
GH200 use the node of the Grace LPDDR memory.

## Method

- SM shares are enforced with CUDA green contexts. A single share uses
  `torch.cuda.GreenContext`; co-runs carve disjoint partitions with
  `partitions.py`.
- Spans are timed with CUDA events around each attention and MoE sublayer
  and summed over all layers. A 200 ms GPU spin before each measured region
  lets the CPU finish enqueueing, so a span measures back-to-back GPU work.
  A 500 us spin before each sublayer, outside the spans, keeps the GPU front
  end's staging window the same at every point.
- A decode point repeats `--decode-runs` times; the maximum and the minimum
  are dropped and the rest averaged. Batched decode feeds fixed random
  tokens instead of greedy output, which keeps expert routing diverse and
  identical across points.
- In a co-run cell the measured side runs eagerly and the contender runs
  as pre-enqueued CUDA graphs that keep its partition busy for the whole
  measured window; each record states whether the contender covered it.
- MoE drivers in co-runs route every token to its top-k experts among a
  fixed set of 104 experts per layer, the mean number of distinct experts a
  4K-token prefill of Qwen3-30B-A3B activates, so they move the bytes a real
  prefill moves.
