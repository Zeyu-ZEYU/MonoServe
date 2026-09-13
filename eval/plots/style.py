"""Shared figure style and data access for the evaluation figures.

The fonts and sizes are those of motivation/plots/style.py. Every series
comes from metrics.sweep() over the results layout (eval/common.py), so the
figures show only what the harness recorded.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import (FixedLocator, FuncFormatter,  # noqa: E402
                               NullFormatter, NullLocator)

from .. import metrics  # noqa: E402
from ..common import DATASETS, find_models, workload_path  # noqa: E402

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
TARGET_LINE = dict(color="#999999", lw=0.7, ls=(0, (1, 1.5)), zorder=2)

# Optional look for the method names of the paper's figures; any other name
# takes the next entry of FALLBACK.
KNOWN = {"MonoServe": ("#1f3a93", "-", "o"), "EP": ("#7f7f7f", "--", "s"),
         "Local": ("#c0392b", "-", "^"), "MuxWise": ("#d4a017", "-", "v"),
         "MonoServe/ML": ("#c0392b", "-", "^"),
         "MonoServe/MCA": ("#d4a017", "-", "v"),
         "MonoServe/KF": ("#2e8b57", "--", "s")}
FALLBACK = [(BLUE, "-", "o"), (ORANGE, "-", "^"), (GREEN, "-", "v"),
            (GRAY, "--", "s"), ("#8e44ad", "-", "D"), ("#17becf", "-.", "P"),
            ("#8c564b", ":", "X")]
TICKS = [round(m * 10.0 ** e, 3) for e in range(-3, 7) for m in (1, 2, 5)]


def method_styles(methods):
    """{method: (color, linestyle, marker)}"""
    out, k = {}, 0
    for m in methods:
        if m in KNOWN:
            out[m] = KNOWN[m]
        else:
            out[m] = FALLBACK[k % len(FALLBACK)]
            k += 1
    return out


def log_axis(axis, lo, hi, max_labels=4):
    """A few labeled major ticks on a log axis and no minor labels."""
    ticks = [t for t in TICKS if lo * 0.999 <= t <= hi * 1.001]
    while len(ticks) > max_labels:     # thin out, keeping the high end
        ticks = ticks[len(ticks) % 2 == 0::2]
    axis.set_major_locator(FixedLocator(ticks or sorted({lo, hi})))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_locator(NullLocator())
    axis.set_minor_formatter(NullFormatter())


def args_parser(description, default_name):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", nargs="+",
                    help="model directories, one row each (default: all)")
    ap.add_argument("--labels", nargs="+", default=[], metavar="MODEL=LABEL",
                    help="display names of the models")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--methods", nargs="+",
                    help="methods in legend order (default: all, sorted)")
    ap.add_argument("--alpha", type=float, default=5.0)
    ap.add_argument("--solo-mode", choices=("exact", "buckets"),
                    default="exact")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--name", default=default_name)
    return ap


def model_label(args, model):
    return dict(s.split("=", 1) for s in args.labels).get(model, model)


def grid(args, alpha=None):
    """(models, datasets, {(model, dataset): metrics.sweep() result}) for
    the models and datasets that have a workload file."""
    models = args.models or find_models(args.results)
    have = {(m, d) for m in models for d in args.datasets
            if workload_path(args.results, m, d).exists()}
    datasets = [d for d in args.datasets if any((m, d) in have for m in models)]
    models = [m for m in models if any((m, d) in have for d in datasets)]
    data = {(m, d): metrics.sweep(args.results, m, d,
                                  args.alpha if alpha is None else alpha,
                                  args.solo_mode, args.methods)
            for m, d in sorted(have)}
    if not data:
        raise SystemExit(f"no workload files under {args.results}")
    return models, datasets, data


def methods_in(args, data):
    found = {m for panel in data.values() for m in panel}
    return [m for m in args.methods if m in found] if args.methods \
        else sorted(found)


def finish_axes(ax):
    ax.grid(True, which="major", lw=0.4, alpha=0.45)
    ax.tick_params(direction="in", length=2.0, which="both", pad=1.5)


def plot_series(ax, xs, ys, look, label):
    color, ls, mk = look
    ax.plot(xs, ys, color=color, ls=ls, marker=mk, mfc="white", mew=0.5,
            label=label)


def legend(fig, axes, ncol):
    """One legend above the figure with every label of every panel."""
    seen = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    fig.legend(list(seen.values()), list(seen), loc="lower center",
               ncol=ncol, bbox_to_anchor=(0.5, 1.0), frameon=False,
               handlelength=1.6, markerscale=1.5, columnspacing=0.9,
               handletextpad=0.4, borderaxespad=0.0)


def save(fig, args):
    out = Path(args.out_dir) / args.name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {out}")
