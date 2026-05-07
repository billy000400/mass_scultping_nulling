#!/usr/bin/env python3
"""Finetuned Pareto: DisCo-only vs S1+DisCo (k=1,2,3), eff@1% vs JSD.

Reads the four per-method ``summary.csv`` files in ``runs/`` and
replots Fig. 4 of the paper. The Pareto-optimal upper envelope of
each method is drawn as a heavy line; off-frontier points are
greyed scatter for visual continuity.

No in-figure title (ICML rule).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# (label, csv-path-relative-to-project-root, color, marker)
DEFAULT_RUNS = [
    ("DisCo-only pretrained",
     "runs/disco_pareto_pretrained/summary.csv",
     "#1f77b4", "o"),
    ("S1+DisCo k=1 ep40",
     "runs/cure_disco_pareto__S1_massshift_ep40_k1/summary.csv",
     "#ff7f0e", "s"),
    ("S1+DisCo k=2 ep40",
     "runs/cure_disco_pareto__S1_massshift_ep40_k2/summary.csv",
     "#2ca02c", "D"),
    ("S1+DisCo k=3 ep40",
     "runs/cure_disco_pareto__S1_massshift_ep40_k3/summary.csv",
     "#d62728", "^"),
]


def _pareto_upper(jsd: np.ndarray, eff: np.ndarray) -> np.ndarray:
    """Return indices that lie on the Pareto upper envelope:
    min JSD for max eff. Sweep by JSD ascending and keep points whose
    eff strictly exceeds the running max-so-far."""
    order = np.argsort(jsd)
    keep = []
    eff_max = -np.inf
    for i in order:
        if eff[i] > eff_max:
            keep.append(i)
            eff_max = eff[i]
    return np.array(keep, dtype=int)


def _read_summary(path: Path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["disco_lambda"] = float(r["disco_lambda"])
            r["eff_at_1pct"]  = float(r["eff_at_1pct"])
            r["jsd"]          = float(r["jsd"])
            rows.append(r)
    return rows


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-jsd", type=float, default=0.024,
                    help="x-axis cap on JSD (default 0.024)")
    ap.add_argument("--out-dir", type=Path, default=here / "results")
    ap.add_argument("--stem", default="pareto_phase4_ep40_final")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for label, rel_path, color, marker in DEFAULT_RUNS:
        path = here / rel_path
        if not path.exists():
            print(f"  WARNING: missing {path}, skipping {label}")
            continue
        rows = _read_summary(path)
        rows = [r for r in rows if r["jsd"] <= args.max_jsd]
        if not rows:
            print(f"  WARNING: no rows under JSD<={args.max_jsd} for {label}")
            continue
        jsd = np.array([r["jsd"]         for r in rows])
        eff = np.array([r["eff_at_1pct"] for r in rows])

        idx = _pareto_upper(jsd, eff)
        # Off-frontier points (light): all rows
        ax.scatter(jsd, eff, color=color, marker=marker, s=30, alpha=0.45,
                   edgecolors="none")
        # Pareto frontier (heavy): connect upper envelope
        ax.plot(jsd[idx], eff[idx], "-", color=color, linewidth=2.0,
                marker=marker, markersize=7, label=label)

        print(f"  {label}: {len(rows)} pts, "
              f"frontier={len(idx)} pts, "
              f"JSD min={jsd.min():.4f}  Eff max={eff.max():.4f}")

    ax.set_xlabel("JSD (QCD mass sculpting, lower is better)")
    ax.set_ylabel("Eff@1% (Hbb efficiency at 1% QCD mis-id, higher is better)")
    # No in-figure title (ICML rule)
    ax.set_xlim(0, args.max_jsd)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
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
