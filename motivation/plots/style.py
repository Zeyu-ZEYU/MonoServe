"""Shared figure style and CSV loading for the Section 2 figures."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Data plotted in the paper (one GH200, Qwen3-30B-A3B-Instruct-2507).
DEFAULT_DATA = Path(__file__).resolve().parent.parent / "data" / "gh200"

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.0,
    "lines.markersize": 2.2,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

GRAY, BLUE, ORANGE, GREEN = "#8A8A8A", "#3B7BB8", "#D07032", "#4A8C5C"
PCT_TICKS = (6, 24, 48, 73, 100)


def read_csv(data_dir, name):
    with open(Path(data_dir) / name, newline="") as f:
        return list(csv.DictReader(f))


def viridis(keys):
    cmap = plt.get_cmap("viridis")
    keys = sorted(keys)
    return {k: cmap(i / max(1, len(keys) - 1)) for i, k in enumerate(keys)}


def pct_ticks(ax, xs):
    labeled = [x for x in sorted(set(xs)) if round(x) in PCT_TICKS]
    ax.set_xticks(labeled)
    ax.set_xticklabels([f"{round(x)}" for x in labeled])


def args_parser(description, default_name):
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data", default=str(DEFAULT_DATA),
                    help="directory with the CSV tables (default: the data "
                         "plotted in the paper)")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--name", default=default_name)
    return ap


def save(fig, args):
    out = Path(args.out_dir) / args.name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
    print(f"wrote {out}")
