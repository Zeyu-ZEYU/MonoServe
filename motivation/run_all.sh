#!/bin/bash
# Run every Section 2 measurement on the local GPU, then build the tables
# and figures from the new results.
#
#   bash motivation/run_all.sh <model-path> [results-dir]
#
# Set LAUNCH to prefix each run, for example to keep the pinned expert
# copies on the CPU socket next to the GPU:
#   LAUNCH="numactl --cpunodebind=0 --membind=0" bash motivation/run_all.sh ...
set -euo pipefail
MODEL=${1:?usage: run_all.sh <model-path> [results-dir]}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${2:-$HERE/results}
LAUNCH=${LAUNCH:-}
mkdir -p "$OUT"
cd "$HERE"

python moe_kernels.py                      # kernel correctness
python partitions.py                       # disjoint SM partitions
$LAUNCH python latency_probe.py | tee "$OUT/latency_probe.txt"
python green_ctx_recarve.py | tee "$OUT/green_ctx_recarve.txt"

# Figs. 2 and 4: SM-share sweeps over input lengths 1K-128K
$LAUNCH python sm_share.py --model-path "$MODEL" --moe-kernel sym \
    --out "$OUT/sm_share_sym.jsonl"
$LAUNCH python sm_share.py --model-path "$MODEL" --moe-kernel asym \
    --phase sweep --skip-decode --out "$OUT/sm_share_asym.jsonl"

# Figs. 13 and 14: batched decode
$LAUNCH python sm_share_batch.py --model-path "$MODEL" \
    --batches 8 16 32 64 128 --moe-kernel sym --sym-block-m 64 \
    --out "$OUT/batch_sym64.jsonl"
$LAUNCH python sm_share_batch.py --model-path "$MODEL" \
    --batches 1 2 4 8 16 32 64 128 256 --moe-kernel sym --sym-block-m auto \
    --decode-runs 1 --skip-validate --out "$OUT/batch_sym_auto.jsonl"
$LAUNCH python sm_share_batch.py --model-path "$MODEL" \
    --batches 1 2 4 8 16 32 64 128 256 --moe-kernel asym \
    --decode-runs 1 --skip-validate --out "$OUT/batch_asym.jsonl"

# Figs. 5, 6, and 15: co-runs on disjoint partitions
for SUITE in pairs reader copy-engine; do
    $LAUNCH python corun.py --model-path "$MODEL" --suite "$SUITE" \
        --out "$OUT/corun_${SUITE//-/_}.jsonl"
done

python plots/extract.py --results "$OUT" --out "$OUT/csv"
for f in plots/fig*.py; do
    python "$f" --data "$OUT/csv" --out-dir "$OUT/figures"
done
