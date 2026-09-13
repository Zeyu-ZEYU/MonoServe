"""Fig. 5: pair-wise co-location of prefill computations at a 4K input
length, each side on a disjoint 48% of the SMs: co-run slowdown of both
sides."""
from matplotlib.patches import Patch

import style
from style import plt

COLOR = {"attention": (style.GRAY, None), "symmetric": (style.ORANGE, "///"),
         "asymmetric": (style.BLUE, None)}


def main():
    args = style.args_parser(__doc__, "fig05_corun_pairs.pdf").parse_args()
    rows = style.read_csv(args.data, "fig05_corun_pairs.csv")
    pairs = list(dict.fromkeys(r["pair"] for r in rows))
    fig, ax = plt.subplots(figsize=(3.35, 1.25))
    w, gap = 0.36, 0.14
    for i, pair in enumerate(pairs):
        bars = [r for r in rows if r["pair"] == pair]
        bars.sort(key=lambda r: r["bar"] != "left")
        for j, r in enumerate(bars):
            v = float(r["corun_slowdown"])
            color, hatch = COLOR[r["computation"]]
            x = i + (j - 0.5) * (w + gap)
            ax.bar(x, v, w, color=color, hatch=hatch, edgecolor="white",
                   lw=0.5, zorder=3)
            ax.text(x, v + 0.09, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=6.5, color="#333333")
    ax.axhline(1.0, color="#999999", lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([p.replace("+", "\n+") for p in pairs])
    ax.set_ylabel("Co-run\nslowdown (x)", ha="center", ma="center")
    ax.set_ylim(0, max(float(r["corun_slowdown"]) for r in rows) + 1.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(fc=style.GRAY, label="Attention"),
                       Patch(fc=style.ORANGE, hatch="///", ec="white",
                             label="SymGEMM"),
                       Patch(fc=style.BLUE, label="AsymGEMM")],
              loc="upper center", frameon=False, ncol=3, handlelength=0.9)
    fig.tight_layout(pad=0.35)
    style.save(fig, args)


if __name__ == "__main__":
    main()
