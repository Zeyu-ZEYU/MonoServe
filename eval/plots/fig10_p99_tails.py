"""Fig. 10: P99 TTFT over the solo TTFT that sets each request's target (top
rows) and P99 TPOT (bottom rows) against the cluster request rate, one row
per model and metric, one column per dataset. Dotted lines: alpha on the
TTFT rows; on the TPOT rows, the median TPOT target of the panel's requests.
Points whose P99 is infinite (more than 1% of the requests did not finish)
or beyond the axis sit at the top edge.

    python -m eval.plots.fig10_p99_tails --results results --out-dir figures

With the ablation variants as --methods it draws Figs. 16 and 17.
"""
import math
import statistics

from ..common import DATASETS
from . import style
from .style import plt

KINDS = {"ttft": ("p99_ttft_ratio", "P99 TTFT (× solo)"),
         "tpot": ("p99_tpot_ms", "P99 TPOT (ms)")}


def target(kind, args, panel):
    if kind == "ttft":
        return args.alpha
    for s in panel.values():
        t = [x.tpot * 1e3 for x in s["targets"].values() if x.tpot]
        return statistics.median(t) if t else None
    return None


def main():
    args = style.args_parser(__doc__.split("\n\n")[0],
                             "fig10_p99_tails.pdf").parse_args()
    models, datasets, data = style.grid(args)
    methods = style.methods_in(args, data)
    looks = style.method_styles(methods)
    rows = [(k, m) for k in KINDS for m in models]
    fig, axes = plt.subplots(len(rows), len(datasets), squeeze=False,
                             layout="constrained",
                             figsize=(3.35, 0.3 + 0.6 * len(rows)))
    for i, (kind, model) in enumerate(rows):
        key, ylabel = KINDS[kind]
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            panel = data.get((model, ds), {})
            line = target(kind, args, panel)
            finite = [p[key] for s in panel.values() for p in s["points"]
                      if p[key] is not None and math.isfinite(p[key])]
            vals = finite + ([line] if line else [])
            lo, hi = (min(vals) / 2, max(vals) * 3) if vals else (0.5, 10)
            top = hi / 1.1
            rates = []
            for m in methods:
                if m not in panel:
                    continue
                pts = [(r, min(p[key], top)) for r, p in
                       zip(panel[m]["rates"], panel[m]["points"])
                       if p[key] is not None]
                if pts:
                    style.plot_series(ax, *zip(*pts), looks[m], m)
                    rates += [r for r, _ in pts]
            if line:
                ax.axhline(line, **style.TARGET_LINE)
            ax.set_yscale("log")
            ax.set_ylim(lo, hi)
            style.log_axis(ax.yaxis, lo, hi, max_labels=3)
            if rates:
                ax.set_xscale("log")
                style.log_axis(ax.xaxis, min(rates), max(rates), 3)
            style.finish_axes(ax)
            if i == 0:
                ax.set_title(DATASETS[ds], pad=1.5)
            if j == 0:
                ax.text(0.04, 0.95, style.model_label(args, model),
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=7)
                if model == models[0]:     # one label per block of rows
                    n = len(models)
                    ax.set_ylabel(ylabel)
                    ax.yaxis.set_label_coords(-0.3, 1 - n / 2 - (n - 1) * 0.04)
    style.legend(fig, axes, 4)
    fig.supxlabel("Request rate (req/s)", fontsize=8)
    style.save(fig, args)


if __name__ == "__main__":
    main()
