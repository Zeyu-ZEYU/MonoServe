"""Table 2: rate (req/s) at 90% attainment against the SLO scale alpha, one
row per model, dataset and method. The same runs are scored with every
alpha. Writes a LaTeX tabular (booktabs and multirow) and prints the table.
A "--" cell never reaches 90%; a cell marked >= is at or above 90% at every
swept rate, so the sweep's highest rate is only a lower bound.

    python -m eval.plots.tab02_alpha --results results --out-dir figures \
        --methods MonoServe MuxWise
"""
import math
from pathlib import Path

from ..common import DATASETS
from . import style


def fmt(v, bound):
    """Two significant digits, keeping trailing zeros (0.30, 1.3, 66)."""
    if v is None:
        return "--"
    digits = max(0, 1 - math.floor(math.log10(v))) if v > 0 else 0
    return (">=" if bound else "") + f"{v:.{digits}f}"


def tex(s):
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("_", r"\_"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def main():
    ap = style.args_parser(__doc__.split("\n\n")[0], "tab02_alpha.tex")
    ap.add_argument("--alphas", type=float, nargs="+", default=[2, 3, 5, 10])
    ap.set_defaults(datasets=["longbench"])
    args = ap.parse_args()
    cells = {}                     # (model, dataset, method) -> [cell]
    for a in args.alphas:
        models, datasets, data = style.grid(args, alpha=a)
        for (model, ds), panel in data.items():
            for m, s in panel.items():
                cells.setdefault((model, ds, m), []).append(
                    fmt(s["goodput"], s["censored"]))
    groups = {}
    for (model, ds, m), row in cells.items():
        groups.setdefault((model, ds), []).append((m, row))
    order = style.methods_in(args, {k: dict(v) for k, v in groups.items()})
    head = " & ".join(f"$\\alpha={a:g}$" for a in args.alphas)
    lines = [f"\\begin{{tabular}}{{@{{}}ll{'r' * len(args.alphas)}@{{}}}}",
             "\\toprule", f" & & {head} \\\\", "\\midrule"]
    text = []
    for g, (model, ds) in enumerate(k for k in sorted(groups)):
        rows = sorted(groups[(model, ds)], key=lambda x: order.index(x[0]))
        name = f"{style.model_label(args, model)}, {DATASETS[ds]}"
        if g:
            lines.append("\\addlinespace[1pt]")
        for k, (m, row) in enumerate(rows):
            first = (f"\\multirow{{{len(rows)}}}{{*}}{{{tex(name)}}}"
                     if k == 0 else "")
            vals = " & ".join(c.replace(">=", "$\\geq$") for c in row)
            lines.append(f"{first} & {tex(m)} & {vals} \\\\")
            text.append(f"{name:<28} {m:<16} " +
                        " ".join(f"{c:>8}" for c in row))
    lines += ["\\bottomrule", "\\end{tabular}"]
    print(f"{'':<28} {'alpha':<16} " +
          " ".join(f"{a:>8g}" for a in args.alphas))
    print("\n".join(text))
    out = Path(args.out_dir) / args.name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
