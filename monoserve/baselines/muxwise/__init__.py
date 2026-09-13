# SPDX-License-Identifier: Apache-2.0
"""The MuxWise baseline: prefill-decode multiplexing on green contexts,
ported onto the Local baseline.

MuxWise's implementation (github.com/ykcombat/sglang, branch slo_config,
commit eeac514; Apache-2.0, Copyright the SGLang project contributors)
runs on SGLang. The port keeps its mechanism and runs it on vLLM with
Local's model and kernels (vLLM's model, FlashAttention, and the Local
plugin's expert placement):

- stream groups: one (prefill, decode) pair of streams per SM division,
  each division two green contexts; one group runs prefill alone and one
  runs decode alone on every SM (config.py, runtime.py);
- one prefill batch at a time, split by layers: each engine step runs
  max(1, split_forward_token_budget // tokens) of its layers on the
  prefill partition while a decode step runs on the decode partition
  (scheduler.py, runtime.py);
- the division is re-chosen when a prefill batch starts, when one
  finishes, and when the decode batch empties: the last division whose
  decode batch-size threshold the decode batch reaches (config.py);
- decode CUDA graphs captured per division and batch size (runtime.py).

The division table comes from the offline profile of the MuxWise paper
(profile.py): decode latency alone on each decode share, decode latency
beside prefills of several sizes on each division, the worst slowdown per
cell, and per decode batch size the division with the fewest decode SMs
whose predicted step meets the TPOT target.

Two defects of the source are fixed: a decode batch below every threshold
left the division unassigned, and a finished prefill batch could be merged
while its kernels were still running.
"""
