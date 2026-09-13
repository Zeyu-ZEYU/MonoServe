"""Fig. 14: decode MoE latency versus SM allocation for the symmetric
(solid) and asymmetric (dashed) kernels (a 1K input length, 128 decode
steps); log y axis, nine batch sizes interleaved across two panels."""
from collections import defaultdict

from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

import style
from style import plt

PANELS = [([1, 4, 16, 64, 256], "(a) B = 1, 4, 16, 64, 256"),
          ([2, 8, 32, 128], "(b) B = 2, 8, 32, 128")]


def main():
    args = style.args_parser(__doc__, "fig14_decode_kernels.pdf").parse_args()
    data = defaultdict(lambda: defaultdict(list))
    for r in style.read_csv(args.data, "fig14_decode_kernel_forms.csv"):
        data[r["kernel"]][int(r["batch"])].append(
            (float(r["sm_pct"]), float(r["moe_latency_s"])))
    batches = sorted(data["symmetric"])
    colors = style.viridis(batches)
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.6), sharex=True,
                             sharey=True)
    handles = {}
    for ax, (bs, sub) in zip(axes, PANELS):
        xs = []
        for B in bs:
            for kern, ls, mk in (("symmetric", "-", "o"),
                                 ("asymmetric", "--", "s")):
                pts = sorted(data[kern].get(B, []))
                if not pts:
                    continue
                x, y = zip(*pts)
                xs += x
                (h,) = ax.plot(x, y, marker=mk, linestyle=ls, color=colors[B])
                if kern == "symmetric":
                    handles[B] = h
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50]))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis="y", which="major", alpha=0.3, linewidth=0.4)
        style.pct_ticks(ax, xs)
        ax.tick_params(length=2, pad=1.5)
        ax.set_xlabel("SM allocation (%)\n" + sub, labelpad=2)
    axes[0].set_ylabel("Total MoE latency (s)")
    keys = [B for B in batches if B in handles]
    fig.legend([handles[B] for B in keys], [str(B) for B in keys],
               loc="lower center", ncol=len(keys), frameon=False,
               title="Batch size (solid: symmetric, dashed: asymmetric)",
               bbox_to_anchor=(0.53, 0.8), handlelength=0.9,
               columnspacing=0.45, handletextpad=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.84], w_pad=0.8)
    style.save(fig, args)


if __name__ == "__main__":
    main()
