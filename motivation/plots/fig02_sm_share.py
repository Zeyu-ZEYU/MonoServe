"""Fig. 2: attention and MoE latency versus SM allocation, one prefill pass
and 128 decode steps at batch size 1, input lengths 1K to 128K."""
import math
from collections import defaultdict

import style
from style import plt


def main():
    args = style.args_parser(__doc__, "fig02_sm_share.pdf").parse_args()
    data = defaultdict(lambda: defaultdict(list))
    for r in style.read_csv(args.data, "fig02_sm_share.csv"):
        data[(r["phase"], r["module"])][int(r["input_len"])].append(
            (float(r["sm_pct"]), float(r["latency_s"])))
    lengths = sorted({L for p in data.values() for L in p})
    colors = style.viridis(lengths)
    panels = [("prefill", "attention", "(a) Prefill: attention"),
              ("prefill", "moe", "(b) Prefill: MoE"),
              ("decode", "attention", "(c) Decode: attention"),
              ("decode", "moe", "(d) Decode: MoE")]
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 2.1), sharex=True)
    for i, (ax, (phase, mod, title)) in enumerate(zip(axes.flat, panels)):
        ymin, xs = math.inf, []
        for L in lengths:
            pts = sorted(data[(phase, mod)].get(L, []))
            if not pts:
                continue
            x, y = zip(*pts)
            xs += x
            ymin = min(ymin, min(y))
            ax.plot(x, y, marker="o", color=colors[L], label=f"{L // 1024}K")
        ax.set_ylim(bottom=0 if phase == "prefill"
                    else max(0.0, math.floor(ymin * 0.99 * 50) / 50))
        ax.grid(axis="y", alpha=0.3, linewidth=0.4)
        style.pct_ticks(ax, xs)
        ax.tick_params(length=2, pad=1.5)
        ax.set_xlabel(title if i < 2 else "SM allocation (%)\n" + title,
                      labelpad=2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Total\nlatency (s)", ha="center", ma="center")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, frameon=False,
               title="Input length", bbox_to_anchor=(0.53, 0.87),
               handlelength=0.9, columnspacing=0.45, handletextpad=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.90], h_pad=0.6, w_pad=0.8)
    style.save(fig, args)


if __name__ == "__main__":
    main()
