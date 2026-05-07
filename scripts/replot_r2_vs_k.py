#!/usr/bin/env python3
"""Replot r2_vs_k from a previously computed CSV (no recomputation).

Reads results/r2_vs_k.csv and writes results/r2_vs_k.pdf and .png with
the publication layout: each method curve anchored at (k=0, r2_base),
linear x-axis from 0 to 16, no symlog gap.

Usage:
  python scripts/replot_r2_vs_k.py
  python scripts/replot_r2_vs_k.py --csv path/to/r2_vs_k.csv \
                                    --out-dir path/to/results
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["S1 (mass-binned PCA)", "RAV (iterative)", "plain PCA", "random"]
METHOD_COLOR = {
    "S1 (mass-binned PCA)": "#1f77b4",
    "RAV (iterative)":      "#ff7f0e",
    "plain PCA":            "#2ca02c",
    "random":               "#7f7f7f",
}


def _read_rows(csv_path: Path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            r["k"] = int(r["k"])
            r["r2_test"] = float(r["r2_test"])
            std = r.get("r2_test_std", "")
            r["r2_test_std"] = float(std) if std not in (None, "", "nan") else 0.0
            rows.append(r)
    return rows


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path,
                    default=here / "results" / "r2_vs_k.csv")
    ap.add_argument("--out-dir", type=Path,
                    default=here / "results")
    ap.add_argument("--stem", default="r2_vs_k")
    args = ap.parse_args()

    rows = _read_rows(args.csv)
    base_rows = [r for r in rows if r["method"] == "baseline" and r["k"] == 0]
    if not base_rows:
        raise SystemExit(f"no baseline (k=0) row in {args.csv}")
    r2_base = base_rows[0]["r2_test"]
    print(f"baseline R^2 = {r2_base:.4f}")

    fig, ax = plt.subplots(figsize=(5.6, 3.8))

    for name in METHOD_ORDER:
        # Anchor each method at (k=0, r2_base): k=0 = no nulling,
        # which is identical across methods by construction.
        ks, r2s, errs = [0], [r2_base], [0.0]
        for r in rows:
            if r["method"] != name or r["k"] == 0:
                continue
            ks.append(r["k"])
            r2s.append(r["r2_test"])
            errs.append(r["r2_test_std"])
        order = np.argsort(ks)
        ks = np.array(ks)[order]
        r2s = np.array(r2s)[order]
        errs = np.array(errs)[order]
        if name == "random" and np.any(errs > 0):
            ax.errorbar(ks, r2s, yerr=errs, fmt="s--",
                        color=METHOD_COLOR[name], lw=1.2, ms=5,
                        capsize=2, label=name)
        else:
            ax.plot(ks, r2s, "o-", color=METHOD_COLOR[name],
                    lw=1.4, ms=5, label=name)

    ax.axhline(r2_base, ls=":", lw=1.0, color="black", alpha=0.7,
               label=f"no nulling ({r2_base:.3f})")
    ax.set_xlabel("k (number of nulled directions)")
    ax.set_ylabel(r"$R^2$ of jet $m_{\mathrm{SD}}$ from $h'$ (linear probe, QCD test)")
    # No in-figure title; the LaTeX caption is the title (ICML rule).
    # Linear x-axis -- no symlog gap between k=0 and k=1.
    ax.set_xticks([0, 1, 2, 4, 8, 16])
    ax.set_xlim(-0.4, 16.4)
    ax.set_ylim(-0.02, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
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
