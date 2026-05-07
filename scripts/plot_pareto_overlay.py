#!/usr/bin/env python3
"""Overlay Pareto frontiers (eff@1% vs JSD) from any number of sweep summary.csv files.

Usage:
  python scripts/plot_pareto_overlay.py \\
      --run "DisCo only:runs/disco_pareto_ce_only/summary.csv" \\
      --run "CURE(QCD-SVD) + DisCo:runs/cure_disco_pareto_ce_only/summary.csv" \\
      --out runs/pareto_overlay.png \\
      [--max-jsd 0.024] [--title ...]

Points are connected by a dotted line (order: ascending JSD) and the *upper Pareto
boundary* (non-dominated set with highest eff@1% at each JSD) is drawn on top as
a thicker solid line. Annotates disco_lambda at each point when --annotate is set.
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                row["eff_at_1pct"] = float(row["eff_at_1pct"])
                row["jsd"] = float(row["jsd"])
                row["disco_lambda"] = float(row["disco_lambda"])
            except (KeyError, ValueError):
                continue
            rows.append(row)
    return sorted(rows, key=lambda r: r["jsd"])


def pareto_upper(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Indices of Pareto-optimal points (min x, max y), sorted by x."""
    order = np.argsort(xs)
    keep = []
    best_y = -np.inf
    for i in order:
        if ys[i] > best_y:
            keep.append(i)
            best_y = ys[i]
    return np.array(keep, dtype=int)


def _plot_one(ax, rows: list[dict], label: str, color, marker: str, annotate: bool,
              y_metric: str = "eff_at_1pct") -> None:
    if not rows:
        return
    rows = [r for r in rows if r.get(y_metric) is not None]
    if not rows:
        return
    jsd = np.array([r["jsd"] for r in rows])
    eff = np.array([float(r[y_metric]) for r in rows])
    lam = np.array([r["disco_lambda"] for r in rows])

    idx = pareto_upper(jsd, eff)
    ax.plot(jsd, eff, ":", color=color, alpha=0.5, linewidth=1)
    ax.scatter(jsd, eff, color=color, marker=marker, s=36, alpha=0.6)
    ax.plot(jsd[idx], eff[idx], "-", color=color, linewidth=2.5,
            marker=marker, markersize=9, label=label)

    if annotate:
        for j, e, l in zip(jsd, eff, lam):
            ax.annotate(f"{l:g}", (j, e), fontsize=7, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")


def plot_overlay(
    runs: list[tuple[str, str]],
    out_path: str,
    *,
    max_jsd: Optional[float] = None,
    title: str = "Pareto frontier: eff@1% vs JSD",
    annotate: bool = False,
    y_metric: str = "eff_at_1pct",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(runs))))
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "h", "<"]

    for i, (label, path) in enumerate(runs):
        if not os.path.isfile(path):
            print(f"  WARNING: missing {path}, skipping {label}")
            continue
        rows = load_summary(path)
        if max_jsd is not None:
            rows = [r for r in rows if r["jsd"] <= max_jsd]
        print(f"  {label}: {len(rows)} points from {path}")
        _plot_one(ax, rows, label, colors[i], markers[i % len(markers)], annotate,
                  y_metric=y_metric)

    y_labels = {
        "eff_at_1pct": "eff@1% (Hbb efficiency at 1% QCD mis-id, higher is better)",
        "auc": "AUC (ROC area, higher is better)",
    }
    ax.set_xlabel("JSD (QCD mass sculpting, lower is better)")
    ax.set_ylabel(y_labels.get(y_metric, y_metric))
    ax.set_title(title)
    if max_jsd is not None:
        ax.set_xlim(0, max_jsd)
    else:
        ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _parse_run(s: str) -> tuple[str, str]:
    if ":" not in s:
        raise argparse.ArgumentTypeError(f"expected 'label:path', got {s!r}")
    label, path = s.split(":", 1)
    return label.strip(), path.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", type=_parse_run, required=True,
                    help="label:path to a sweep summary.csv (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-jsd", type=float, default=None)
    ap.add_argument("--title", default="Pareto frontier: eff@1% vs JSD")
    ap.add_argument("--annotate", action="store_true",
                    help="Annotate each point with its disco_lambda")
    ap.add_argument("--y-metric", default="eff_at_1pct",
                    help="Column to use as y-axis (default: eff_at_1pct; also: auc)")
    args = ap.parse_args()

    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs = [(lab, p if os.path.isabs(p) else os.path.join(proj, p))
            for lab, p in args.run]
    out = args.out if os.path.isabs(args.out) else os.path.join(proj, args.out)
    plot_overlay(runs, out, max_jsd=args.max_jsd, title=args.title, annotate=args.annotate,
                 y_metric=args.y_metric)


if __name__ == "__main__":
    main()
