"""Turn measurement JSONL files into the CSV tables the figure scripts read.

    python plots/extract.py --results motivation/results --out motivation/results/csv

Files expected in --results (a missing file is skipped):
  sm_share_sym.jsonl        sm_share.py --moe-kernel sym           Figs. 2, 4
  sm_share_asym.jsonl       sm_share.py --moe-kernel asym          Fig. 4
  batch_sym64.jsonl         sm_share_batch.py --sym-block-m 64     Fig. 13
  batch_sym_auto.jsonl      sm_share_batch.py --sym-block-m auto   Fig. 14
  batch_asym.jsonl          sm_share_batch.py --moe-kernel asym    Fig. 14
  corun_pairs.jsonl         corun.py --suite pairs                 Fig. 5
  corun_reader.jsonl        corun.py --suite reader                Fig. 6
  corun_copy_engine.jsonl   corun.py --suite copy-engine           Fig. 15

The tables have the same columns as the ones in data/gh200.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def val_s(m):
    return m.get("value", m["mean"]) / 1e3


def write(out, name, header, recs):
    with open(out / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(recs)
    print(f"{name}: {len(recs)} rows")


def slowdown(rs, cell, side):
    solo = corun = None
    for r in rs:
        if r["cell"] == cell and r["side"] == side:
            if r["mode"] == "solo":
                solo = statistics.median(r["times_ms"])
            else:
                corun = statistics.median(r["times_ms"])
    return None if solo is None or corun is None else corun / solo


PAIRS = [("Attn+Attn", "attn+attn", "attention", "attention-2"),
         ("Sym+Sym", "sym+sym", "sym", "sym-2"),
         ("Asym+Asym", "asym+asym", "asym", "asym-2"),
         ("Sym+Asym", "sym+asym", "sym-2", "asym"),
         ("Attn+Sym", "attn+sym", "attention", "sym"),
         ("Attn+Asym", "attn+asym", "attention", "asym")]
KIND = {"attention": "attention", "attention-2": "attention",
        "sym": "symmetric", "sym-2": "symmetric",
        "asym": "asymmetric", "asym-2": "asymmetric"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def have(name):
        ok = (res / name).exists()
        if not ok:
            print(f"skip: {name} not found")
        return ok

    if have("sm_share_sym.jsonl"):
        recs = []
        for r in rows(res / "sm_share_sym.jsonl"):
            for mod, key in (("attention", "attn_ms"), ("moe", "moe_ms")):
                recs.append([r["phase"], mod, r["L"], r["sm"],
                             round(r["sm_pct"], 3), f"{val_s(r[key]):.6g}"])
        write(out, "fig02_sm_share.csv",
              ["phase", "module", "input_len", "sms", "sm_pct", "latency_s"],
              recs)
        if have("sm_share_asym.jsonl"):
            recs = []
            for kern, fn in (("symmetric", "sm_share_sym.jsonl"),
                             ("asymmetric", "sm_share_asym.jsonl")):
                for r in rows(res / fn):
                    if r["phase"] == "prefill":
                        recs.append([kern, r["L"], r["sm"],
                                     round(r["sm_pct"], 3),
                                     f"{val_s(r['moe_ms']):.6g}"])
            write(out, "fig04_prefill_kernel_forms.csv",
                  ["kernel", "input_len", "sms", "sm_pct", "moe_latency_s"],
                  recs)

    if have("batch_sym64.jsonl"):
        recs = []
        for r in rows(res / "batch_sym64.jsonl"):
            for mod, key in (("attention", "attn_ms"), ("moe", "moe_ms")):
                recs.append([mod, r["B"], r["sm"], round(r["sm_pct"], 3),
                             f"{val_s(r[key]):.6g}"])
        write(out, "fig13_batched_decode.csv",
              ["module", "batch", "sms", "sm_pct", "latency_s"], recs)

    if have("batch_sym_auto.jsonl") and have("batch_asym.jsonl"):
        recs = []
        for kern, fn in (("symmetric", "batch_sym_auto.jsonl"),
                         ("asymmetric", "batch_asym.jsonl")):
            for r in rows(res / fn):
                recs.append([kern, r["B"], r["sm"], round(r["sm_pct"], 3),
                             f"{val_s(r['moe_ms']):.6g}"])
        write(out, "fig14_decode_kernel_forms.csv",
              ["kernel", "batch", "sms", "sm_pct", "moe_latency_s"], recs)

    if have("corun_pairs.jsonl"):
        rs = rows(res / "corun_pairs.jsonl")
        recs = []
        for label, cell, left, right in PAIRS:
            for bar, side in (("left", left), ("right", right)):
                v = slowdown(rs, cell, side)
                if v is not None:
                    recs.append([label, bar, KIND[side], f"{v:.4f}"])
        write(out, "fig05_corun_pairs.csv",
              ["pair", "bar", "computation", "corun_slowdown"], recs)

    if have("corun_reader.jsonl"):
        rs = rows(res / "corun_reader.jsonl")
        recs = []
        for place, label in (("cpu", "cpu_dram"), ("hbm", "hbm")):
            cells = sorted({r["cell"] for r in rs
                            if r["cell"].startswith(f"attn+reader-{place}-")},
                           key=lambda c: int(c.rsplit("-", 1)[1]))
            for cell in cells:
                v = slowdown(rs, cell, "attention")
                if v is not None:
                    recs.append([label, int(cell.rsplit("-", 1)[1]),
                                 f"{v:.4f}"])
        write(out, "fig06_reader_placement.csv",
              ["weights_in", "reader_sms", "attention_slowdown"], recs)

    if have("corun_copy_engine.jsonl"):
        rs = rows(res / "corun_copy_engine.jsonl")
        cells = sorted({r["cell"] for r in rs},
                       key=lambda c: int(c.rsplit("-", 1)[1][:-2]))
        recs = []
        for cell in cells:
            v = slowdown(rs, cell, "attention")
            if v is not None:
                recs.append([int(cell.rsplit("-", 1)[1][:-2]), f"{v:.4f}"])
        write(out, "fig15_copy_engine.csv",
              ["cpu_dram_walked_gb", "attention_slowdown"], recs)


if __name__ == "__main__":
    main()
