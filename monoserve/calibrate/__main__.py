"""Measure a calibration of the time estimator on this GPU.

    python -m monoserve.calibrate --model /path/to/model --out calibration.json

The model directory is read for its shape only (config.json); the runs use
random weights of that shape. --experts limits how many experts per layer
the sweeps allocate (all by default); --quick runs a short version for
smoke tests.
"""
import argparse

from monoserve.calibrate import Calibrator
from monoserve.control import save_calibration
from monoserve.runtime.config import MoEConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="model directory (config.json)")
    ap.add_argument("--out", required=True, help="calibration file to write")
    ap.add_argument("--experts", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    from transformers import AutoConfig
    cfg = MoEConfig.from_hf(AutoConfig.from_pretrained(args.model))
    data = Calibrator(cfg, quick=args.quick, experts=args.experts).run()
    save_calibration(data, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
