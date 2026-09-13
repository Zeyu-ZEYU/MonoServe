"""Fig. 12: rate at 90% attainment against the hot tier's share of the
expert set, relative to the rightmost point (the default tier), one panel
per model and one line per dataset.

Each point is one sweep, a method whose run headers carry meta.hot_share,
the tier's share of the expert set in percent (run_sweep.py --meta
hot_share=41.2). --methods restricts the sweeps considered. A sweep that
never reaches 90% has no rate to plot and is reported and left out.

    python -m eval.plots.fig12_hot_tier --results results --out-dir figures
"""
from ..common import DATASETS
from . import style
from .style import plt

LOOK = {"sharegpt": (style.BLUE, "-", "o"),
        "longbench": (style.ORANGE, "--", "^")}


def main():
    args = style.args_parser(__doc__.split("\n\n")[0],
                             "fig12_hot_tier.pdf").parse_args()
    models, datasets, data = style.grid(args)
    fig, axes = plt.subplots(1, len(models), sharey=True, squeeze=False,
                             layout="constrained",
                             figsize=(3.35, 1.1))
    for ax, model in zip(axes[0], models):
        for ds in datasets:
            pts = []
            for m, s in data.get((model, ds), {}).items():
                share = s["headers"][0].get("meta", {}).get("hot_share")
                if share is None:
                    continue
                if s["goodput"] is None:
                    print(f"{model} {ds} {m}: never reaches 90%, left out")
                    continue
                if s["censored"]:
                    print(f"{model} {ds} {m}: at or above 90% at every "
                          f"rate; its rate is a lower bound")
                pts.append((float(share), s["goodput"]))
            if not pts:
                continue
            pts.sort()
            ref = pts[-1][1]
            style.plot_series(ax, [x for x, _ in pts],
                              [g / ref for _, g in pts],
                              LOOK.get(ds, style.FALLBACK[0]), DATASETS[ds])
        ax.set_title(style.model_label(args, model), pad=1.5)
        ax.set_ylim(bottom=0)
        style.finish_axes(ax)
    axes[0, 0].set_ylabel("Rate at 90% SLO\n(rel. to default)")
    style.legend(fig, axes, 2)
    fig.supxlabel("Hot tier (% of the expert set)", fontsize=8)
    style.save(fig, args)


if __name__ == "__main__":
    main()
