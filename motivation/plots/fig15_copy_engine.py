"""Fig. 15: attention slowdown under a copy-engine transfer reading 1 GB to
128 GB of CPU DRAM."""
import style
from style import plt


def main():
    args = style.args_parser(__doc__, "fig15_copy_engine.pdf").parse_args()
    rows = style.read_csv(args.data, "fig15_copy_engine.csv")
    pts = sorted((int(r["cpu_dram_walked_gb"]), float(r["attention_slowdown"]))
                 for r in rows)
    x, y = zip(*pts)
    fig, ax = plt.subplots(figsize=(3.35, 0.95))
    ax.plot(x, y, "-o", color=style.GREEN, lw=1.1, zorder=3)
    for xi, yi in pts:
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=6, color="#333333")
    ax.axhline(1.0, color="#999999", lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in x])
    ax.minorticks_off()
    ax.set_xlabel("CPU DRAM walked by the copy engine (GB)")
    ax.set_ylabel("Slowdown (x)")
    ax.set_ylim(0.95, max(1.15, max(y) + 0.05))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.45)
    style.save(fig, args)


if __name__ == "__main__":
    main()
