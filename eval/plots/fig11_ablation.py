"""Fig. 11: SLO attainment of the ablation variants against the cluster
request rate, in the form of Fig. 9. Dotted line: 90%. Name the full system
and its variants with --methods, in legend order.

    python -m eval.plots.fig11_ablation --results results --out-dir figures \
        --methods MonoServe MonoServe/ML MonoServe/MCA MonoServe/KF

fig10_p99_tails.py with the same --methods draws the tails behind these
curves (Figs. 16 and 17).
"""
from . import style
from .fig09_attainment import draw


def main():
    draw(style.args_parser(__doc__.split("\n\n")[0],
                           "fig11_ablation.pdf").parse_args(), ncol=2)


if __name__ == "__main__":
    main()
