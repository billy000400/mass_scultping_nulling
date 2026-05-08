#!/usr/bin/env python3
"""Pareto comparison plot: upper envelope per method, no scatter.

Sister to replot_pareto_finetuned.py — same input CSVs but a cleaner
visual: only the Pareto upper envelope of each sweep is drawn as a
heavy line, with no faded off-frontier scatter. Uses a black-and-RGB
palette and an in-figure title to match the previous hand-made
``pareto_actual_comparison.{pdf,png}``.

Default x-axis is linear over the new threshold-based JSD range
(0.018–0.06). Pass ``--log-x`` to recover the original log layout.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# (label, csv-path-relative-to-project-root, color, marker)
DEFAULT_RUNS = [
    ("DisCo-only",
     "runs/disco_pareto_pretrained/summary.csv",
     "black", "o"),
    ("S1+DisCo k=1",
     "runs/cure_disco_pareto__S1_massshift_ep40_k1/summary.csv",
     "#1f77b4", "s"),
    ("S1+DisCo k=2",
     "runs/cure_disco_pareto__S1_massshift_ep40_k2/summary.csv",
     "#ff7f0e", "s"),
    ("S1+DisCo k=3",
     "runs/cure_disco_pareto__S1_massshift_ep40_k3/summary.csv",
     "#2ca02c", "s"),
]


def _read_summary(path: Path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["disco_lambda"] = float(r["disco_lambda"])
            r["eff_at_1pct"] = float(r["eff_at_1pct"])
            r["jsd"] = float(r["jsd"])
            rows.append(r)
    return rows


def _pareto_upper(jsd: np.ndarray, eff: np.ndarray) -> np.ndarray:
    """Upper-envelope indices: sweep by ascending JSD, keep points whose
    eff strictly exceeds the running max-so-far."""
    order = np.argsort(jsd)
    keep = []
    eff_max = -np.inf
    for i in order:
        if eff[i] > eff_max:
            keep.append(i)
            eff_max = eff[i]
    return np.array(keep, dtype=int)


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-jsd", type=float, default=0.008)
    ap.add_argument("--max-jsd", type=float, default=0.05)
    ap.add_argument("--log-x", action="store_true",
                    help="Log x-axis (original style; narrow dynamic range "
                         "under the new metric so linear is the default).")
    ap.add_argument("--no-title", action="store_true",
                    help="Drop the in-figure title (e.g. for ICML camera-ready).")
    ap.add_argument("--out-dir", type=Path, default=here / "results")
    ap.add_argument("--stem", default="pareto_actual_comparison")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    for label, rel_path, color, marker in DEFAULT_RUNS:
        path = here / rel_path
        if not path.exists():
            print(f"  WARNING: missing {path}, skipping {label}")
            continue
        rows = _read_summary(path)
        rows = [r for r in rows
                if args.min_jsd <= r["jsd"] <= args.max_jsd]
        if not rows:
            print(f"  WARNING: no rows in JSD window for {label}")
            continue
        jsd_all = np.array([r["jsd"] for r in rows])
        eff_all = np.array([r["eff_at_1pct"] for r in rows])
        idx = _pareto_upper(jsd_all, eff_all)
        jsd = jsd_all[idx]
        eff = eff_all[idx]
        ax.plot(jsd, eff, "-", color=color, linewidth=1.8,
                marker=marker, markersize=6, label=label)
        print(f"  {label}: {len(rows)} pts → frontier {len(idx)} pts, "
              f"JSD ∈ [{jsd.min():.4f}, {jsd.max():.4f}]  "
              f"Eff ∈ [{eff.min():.4f}, {eff.max():.4f}]")

    ax.set_xlabel("JSD (lower = better)")
    ax.set_ylabel("Eff@1%")
    if not args.no_title:
        ax.set_title("Pareto frontier comparison (Phase 4 ep40)")
    if args.log_x:
        ax.set_xscale("log")
    else:
        ax.set_xlim(args.min_jsd, args.max_jsd)
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
