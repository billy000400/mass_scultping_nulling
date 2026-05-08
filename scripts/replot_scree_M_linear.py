#!/usr/bin/env python3
"""Linear-scale scree plot of the centred mass-bin mean matrix M.

Reads the cached results/scree_values.csv (produced by
scripts/replot_scree_M.py) and produces a linear-y plot showing
per-mode Frobenius-energy share sigma_i^2 / sum_j sigma_j^2 as bars,
with cumulative energy overlaid as a step line. The leading-mode
elbow is visible on a linear scale, unlike the log-scale version.

Usage:
  python scripts/replot_scree_M_linear.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    csv_path = here / "results" / "scree_values.csv"
    out_dir = here / "results"
    stem = "scree_M_linear"

    sigmas: list[float] = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            sigmas.append(float(row["sigma_i"]))
    S = np.asarray(sigmas)

    # Drop numerically-zero tail
    keep = S / S[0] > 1e-6
    S = S[keep]
    energy = S ** 2
    share = energy / energy.sum()
    cum = np.cumsum(share)
    idx = np.arange(1, share.size + 1)

    print(f"per-mode share (k=1..6): {[f'{s:.4f}' for s in share[:6]]}")
    print(f"cumulative      (k=1..6): {[f'{c:.4f}' for c in cum[:6]]}")

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bars = ax.bar(idx, share, color="#1f77b4", alpha=0.85,
                  edgecolor="white", linewidth=0.6,
                  label=r"per-mode share $\sigma_i^2/\sum_j\sigma_j^2$")
    ax2 = ax.twinx()
    ax2.plot(idx, cum, "o-", color="#d62728", lw=1.3, ms=4,
             label="cumulative")
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("cumulative Frobenius-energy fraction",
                   color="#d62728")
    ax2.tick_params(axis="y", colors="#d62728")
    ax2.spines["right"].set_color("#d62728")

    # Annotate cumulative at k=1,2,3
    for k in (1, 2, 3):
        ax2.annotate(f"{cum[k - 1] * 100:.1f}%",
                     xy=(k, cum[k - 1]),
                     xytext=(6, -10), textcoords="offset points",
                     color="#d62728", fontsize=8)

    ax.set_xlabel("singular-value index $i$")
    ax.set_ylabel(r"per-mode share $\sigma_i^2/\sum_j\sigma_j^2$",
                  color="#1f77b4")
    ax.tick_params(axis="y", colors="#1f77b4")
    ax.set_xticks(idx[::2])
    ax.set_ylim(0, max(share) * 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False,
              fontsize=8)

    fig.tight_layout()
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {pdf}")
    print(f"saved {png}")


if __name__ == "__main__":
    main()
