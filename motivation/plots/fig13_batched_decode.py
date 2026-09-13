"""Fig. 13: batched decode, attention and MoE latency versus SM allocation
for batch sizes 8 to 128 (a 1K input length, 128 decode steps)."""
import math
from collections import defaultdict

import style
from style import plt


def main():
    args = style.args_parser(__doc__, "fig13_batched_decode.pdf").parse_args()
    data = defaultdict(lambda: defaultdict(list))
    for r in style.read_csv(args.data, "fig13_batched_decode.csv"):
        data[r["module"]][int(r["batch"])].append(
            (float(r["sm_pct"]), float(r["latency_s"])))
    batches = sorted(data["attention"])
    colors = style.viridis(batches)
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.45), sharex=True)
    for ax, (mod, title) in zip(axes, (("attention", "(a) Decode: attention"),
                                       ("moe", "(b) Decode: MoE"))):
        ymin, xs = math.inf, []
        for B in batches:
            x, y = zip(*sorted(data[mod][B]))
            xs += x
            ymin = min(ymin, min(y))
            ax.plot(x, y, marker="o", color=colors[B], label=f"{B}")
        ax.set_ylim(bottom=max(0.0, math.floor(ymin * 0.99 * 50) / 50))
        ax.grid(axis="y", alpha=0.3, linewidth=0.4)
        style.pct_ticks(ax, xs)
        ax.tick_params(length=2, pad=1.5)
        ax.set_xlabel("SM allocation (%)\n" + title, labelpad=2)
    axes[0].set_ylabel("Total latency (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(batches),
               frameon=False, title="Batch size", bbox_to_anchor=(0.53, 0.78),
               handlelength=0.9, columnspacing=0.45, handletextpad=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.85], w_pad=0.8)
    style.save(fig, args)


if __name__ == "__main__":
    main()
