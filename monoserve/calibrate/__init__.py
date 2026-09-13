"""Calibration of the time estimator, measured inside the fabric.

The estimator's inputs (Section 3.1 of the paper) come from runs of the
fabric's own tiles as lanes on chosen SM counts, timed with the fabric's
in-kernel timers:

  solo sweeps    each kernel alone at every SM share, with its weights in
                 HBM: a compute-bound shape gives its FLOP rate, a
                 read-bound shape its HBM read rate, and the read rate at
                 the smallest share the MTQ entries one SM keeps in flight;
  credit sweeps  per kernel form, the form's DRAM-expert tiles stream from
                 CPU DRAM with a growing number of link slots while the
                 heaviest HBM-reading kernel fills the other SMs: the link
                 rate against credits (rho) and the entries one tile keeps
                 in flight (q);
  probe          a pointer-chasing probe tile gives the residency of an HBM
                 miss (R_H) and of a link miss (R_C) alone, and the stretch
                 gamma of an HBM miss while other lanes load the MTQ to
                 several exposures;
  overheads      decode steps of one request through real layer programs
                 of the model's shape (synthetic weights, one and two MoE
                 layers), less what the estimator's windows predict: the
                 per-layer overhead and the per-pass tail; and, for
                 reference, the latency of one stage transition.

    python -m monoserve.calibrate --model /path/to/model --out calibration.json
"""
from monoserve.calibrate.calibrator import Calibrator

__all__ = ["Calibrator"]
