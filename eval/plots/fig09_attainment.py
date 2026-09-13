"""Fig. 9: SLO attainment against the cluster request rate, one panel per
model (rows) and dataset (columns), one curve per method. Dotted line: 90%.

    python -m eval.plots.fig09_attainment --results results --out-dir figures
"""
from ..common import DATASETS
from . import style
from .style import plt


def draw(args, ncol=4):
    models, datasets, data = style.grid(args)
    methods = style.methods_in(args, data)
    looks = style.method_styles(methods)
    fig, axes = plt.subplots(len(models), len(datasets), sharey=True,
                             squeeze=False, layout="constrained",
                             figsize=(3.35, 0.3 + 0.62 * len(models)))
    for i, model in enumerate(models):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            panel = data.get((model, ds), {})
            rates = []
            for m in methods:
                if m not in panel:
                    continue
                pts = [(r, 100 * p["attainment"]) for r, p in
                       zip(panel[m]["rates"], panel[m]["points"])
                       if p["attainment"] is not None]
                if pts:
                    style.plot_series(ax, *zip(*pts), looks[m], m)
                    rates += [r for r, _ in pts]
            ax.axhline(90, **style.TARGET_LINE)
            ax.set_ylim(0, 148)      # room above 100% for the model name
            ax.set_yticks([0, 50, 100])
            if rates:
                ax.set_xscale("log")
                style.log_axis(ax.xaxis, min(rates), max(rates))
            style.finish_axes(ax)
            if i == 0:
                ax.set_title(DATASETS[ds], pad=1.5)
            if j == 0:
                ax.text(0.04, 0.97, style.model_label(args, model),
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=7)
    style.legend(fig, axes, ncol)
    fig.supxlabel("Request rate (req/s)", fontsize=8)
    fig.supylabel("SLO attainment (%)", fontsize=8)
    style.save(fig, args)


def main():
    draw(style.args_parser(__doc__.split("\n\n")[0],
                           "fig09_attainment.pdf").parse_args())


if __name__ == "__main__":
    main()
