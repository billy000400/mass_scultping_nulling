#!/usr/bin/env python3
"""Frozen-head alpha-sweep Pareto comparison across direction sources.

Reads results/pareto_frozen_methods_alpha_sweep.csv and replots the
no-finetune Pareto for the four direction sources (S1, RAV, plain
PCA, unconstrained baseline) at fixed k=1, alpha sweep in [0, 1].

No in-figure title (ICML rule).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Order, color, marker, line style for each method as it appears in the CSV
METHOD_STYLE = {
    "S1 (mass-binned PCA)": ("#1f77b4", "o", "-",  "S1 (mass-binned PCA)"),
    "plain PCA":            ("#2ca02c", "s", "--", "plain PCA"),
    "RAV (unit OLS)":       ("#ff7f0e", "^", ":",  "RAV (unit OLS)"),
}

UNCONSTRAINED_KEY = "unconstrained_ParT"


def _read_rows(csv_path: Path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            r["alpha"]       = float(r["alpha"])
            r["eff_at_1pct"] = float(r["eff_at_1pct"])
            r["jsd"]         = float(r["jsd"])
            rows.append(r)
    return rows


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path,
                    default=here / "results" /
                            "pareto_frozen_methods_alpha_sweep.csv")
    ap.add_argument("--out-dir", type=Path,
                    default=here / "results")
    ap.add_argument("--stem", default="pareto_frozen_methods_alpha_sweep")
    args = ap.parse_args()

    rows = _read_rows(args.csv)

    # Unconstrained baseline anchor (alpha=0 row)
    base = next(
        (r for r in rows
         if r["method"] == UNCONSTRAINED_KEY and r["alpha"] == 0.0),
        None,
    )
    if base is None:
        # fall back to any method's alpha=0 (they all coincide with baseline)
        base = next(r for r in rows if r["alpha"] == 0.0)
    jsd0, eff0 = base["jsd"], base["eff_at_1pct"]
    print(f"baseline: JSD={jsd0:.4f}  Eff@1%={eff0:.4f}")

    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    for method, (color, marker, ls, label) in METHOD_STYLE.items():
        sub = sorted(
            (r for r in rows if r["method"] == method),
            key=lambda r: r["alpha"],
        )
        if not sub:
            print(f"  WARNING: no rows for {method}, skipping")
            continue
        jsds = np.array([r["jsd"]         for r in sub])
        effs = np.array([r["eff_at_1pct"] for r in sub])

        # If all (jsd, eff) collapse to baseline (RAV does this), draw a
        # single x-marker at the baseline corner instead of an overlapping line.
        spread = np.max(np.abs(jsds - jsd0)) + np.max(np.abs(effs - eff0))
        if spread < 1e-4:
            ax.scatter([jsd0], [eff0], marker="x", s=80, color=color,
                       zorder=4,
                       label=f"{label} (k=1)  α-sweep collapses to baseline")
        else:
            ax.plot(jsds, effs, marker + ls, color=color, lw=1.4, ms=5,
                    label=f"{label} (k=1)")

    # Anchor the unconstrained ParT baseline as a star
    ax.scatter([jsd0], [eff0], marker="*", s=160, color="black",
               zorder=5, label="unconstrained ParT (α=0)")

    ax.set_xlabel("JSD (mass sculpting; lower = better)")
    ax.set_ylabel("Eff@1% (Hbb tag eff at 1% QCD mis-id; higher = better)")
    # No in-figure title (ICML rule)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
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
