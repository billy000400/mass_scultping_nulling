#!/usr/bin/env python3
"""Frozen-head k × alpha-sweep Pareto (appendix figure).

Reads results/pareto_frozen_alpha_sweep.csv and replots the no-finetune
S1 frozen Pareto across k ∈ {1,2,3,...} with alpha sweep in [0, 1].
Pairs with replot_pareto_frozen_methods.py — same style, no in-figure title.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Same color palette as the embedded plot in pareto_frozen_alpha_sweep.py
K_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#9467bd",
    5: "#8c564b",
}


def _read_rows(csv_path: Path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            r["k"] = int(r["k"])
            r["alpha"] = float(r["alpha"])
            r["eff_at_1pct"] = float(r["eff_at_1pct"])
            r["jsd"] = float(r["jsd"])
            rows.append(r)
    return rows


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path,
                    default=here / "results" / "pareto_frozen_alpha_sweep.csv")
    ap.add_argument("--out-dir", type=Path, default=here / "results")
    ap.add_argument("--stem", default="pareto_frozen_alpha_sweep")
    ap.add_argument("--annotate-every", type=int, default=2,
                    help="Annotate every Nth alpha point (default 2)")
    args = ap.parse_args()

    rows = _read_rows(args.csv)

    # Baseline anchor: k=0 row (or any alpha=0 row — they all coincide).
    base = next((r for r in rows if r["k"] == 0), None)
    if base is None:
        base = next(r for r in rows if r["alpha"] == 0.0)
    jsd0, eff0 = base["jsd"], base["eff_at_1pct"]
    print(f"baseline: JSD={jsd0:.4f}  Eff@1%={eff0:.4f}")

    ks = sorted({r["k"] for r in rows if r["k"] > 0})

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for k in ks:
        sub = sorted(
            [r for r in rows if r["k"] == k],
            key=lambda r: r["alpha"],
        )
        if not sub:
            print(f"  WARNING: no rows for k={k}")
            continue
        jsds = np.array([r["jsd"] for r in sub])
        effs = np.array([r["eff_at_1pct"] for r in sub])
        ax.plot(jsds, effs, "o-", color=K_COLORS.get(k, "gray"),
                lw=1.4, ms=5, label=f"S1 frozen, k={k}")
        for i, r in enumerate(sub):
            if i % args.annotate_every == 0:
                ax.annotate(f"α={r['alpha']:.1f}",
                            (r["jsd"], r["eff_at_1pct"]),
                            fontsize=7, alpha=0.6,
                            xytext=(3, 3), textcoords="offset points")

    ax.scatter([jsd0], [eff0], marker="*", s=140, color="black",
               zorder=5, label="unconstrained ParT (α=0)")
    ax.set_xlabel("JSD (mass sculpting; lower = better)")
    ax.set_ylabel("Eff@1% (Hbb tag eff at 1% QCD mis-id; higher = better)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf = args.out_dir / f"{args.stem}.pdf"
    png = args.out_dir / f"{args.stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {pdf}")
    print(f"saved {png}")


if __name__ == "__main__":
    main()
