"""Fig. 4: prefill MoE latency versus SM allocation for the symmetric
(solid) and asymmetric (dashed) kernels; log y axis, eight input lengths
interleaved across two panels."""
from collections import defaultdict

from matplotlib.ticker import FixedLocator

import style
from style import plt

PANELS = [([1024, 4096, 16384, 65536], "(a) 1K, 4K, 16K, 64K"),
          ([2048, 8192, 32768, 131072], "(b) 2K, 8K, 32K, 128K")]


def main():
    args = style.args_parser(__doc__, "fig04_prefill_kernels.pdf").parse_args()
    data = defaultdict(lambda: defaultdict(list))
    for r in style.read_csv(args.data, "fig04_prefill_kernel_forms.csv"):
        data[r["kernel"]][int(r["input_len"])].append(
            (float(r["sm_pct"]), float(r["moe_latency_s"])))
    lengths = sorted({L for d in data.values() for L in d})
    colors = style.viridis(lengths)
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.6), sharex=True,
                             sharey=True)
    handles = {}
    for ax, (lens, sub) in zip(axes, PANELS):
        xs = []
        for L in lens:
            for kern, ls, mk in (("symmetric", "-", "o"),
                                 ("asymmetric", "--", "s")):
                pts = sorted(data[kern].get(L, []))
                if not pts:
                    continue
                x, y = zip(*pts)
                xs += x
                (h,) = ax.plot(x, y, marker=mk, linestyle=ls,
                               color=colors[L], label=f"{L // 1024}K")
                if kern == "symmetric":
                    handles[L] = h
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
        ax.grid(axis="y", which="major", alpha=0.3, linewidth=0.4)
        style.pct_ticks(ax, xs)
        ax.tick_params(length=2, pad=1.5)
        ax.set_xlabel("SM allocation (%)\n" + sub, labelpad=2)
    axes[0].set_ylabel("Total MoE\nlatency (s)", ha="center", ma="center")
    keys = [L for L in lengths if L in handles]
    fig.legend([handles[L] for L in keys], [f"{L // 1024}K" for L in keys],
               loc="lower center", ncol=8, frameon=False,
               title="Input length (solid: symmetric, dashed: asymmetric)",
               bbox_to_anchor=(0.53, 0.8), handlelength=0.9,
               columnspacing=0.45, handletextpad=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.84], w_pad=0.8)
    style.save(fig, args)


if __name__ == "__main__":
    main()
