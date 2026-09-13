"""Fig. 6: attention slowdown under a synthetic expert reader, the same
expert weights placed in CPU DRAM (read over the link) or in HBM."""
import style
from style import plt


def main():
    args = style.args_parser(__doc__, "fig06_reader.pdf").parse_args()
    rows = style.read_csv(args.data, "fig06_reader_placement.csv")
    fig, ax = plt.subplots(figsize=(3.35, 1.3))
    for place, color, label in (("cpu_dram", style.ORANGE,
                                 "Weights in CPU DRAM"),
                                ("hbm", style.BLUE, "Weights in HBM")):
        pts = sorted((int(r["reader_sms"]), float(r["attention_slowdown"]))
                     for r in rows if r["weights_in"] == place)
        if not pts:
            continue
        x, y = zip(*pts)
        ax.plot(x, y, "-o", color=color, label=label, lw=1.1, zorder=3)
        for xi, yi in zip(x, y):
            if yi >= 1.2:
                ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                            xytext=(-3, 3), ha="right", fontsize=6.5,
                            color="#333333")
        ax.set_xticks(x)
    ax.axhline(1.0, color="#999999", lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.set_xlabel("Number of SMs allocated to the reader")
    ax.set_ylabel("Attention\nslowdown (x)", ha="center", ma="center")
    ax.set_ylim(bottom=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout(pad=0.45)
    style.save(fig, args)


if __name__ == "__main__":
    main()
