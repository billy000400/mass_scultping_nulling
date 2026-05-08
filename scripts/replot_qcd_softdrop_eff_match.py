#!/usr/bin/env python3
"""Replot qcd_softdrop_histograms_eff_match{,_eff932_k1}.pdf from CSV.

Reads the two long-format CSVs already produced by
``qcd_softdrop_histograms_eff_match.py`` and rebuilds the figures with
the same colour palette as ``pareto_actual_comparison.pdf``:

    DisCo-only    -> black
    S1+DisCo k=1  -> #1f77b4 (blue)
    S1+DisCo k=2  -> #ff7f0e (orange)

To avoid colliding with DisCo-only, inclusive QCD is drawn in mid-grey
rather than black.

No in-figure title (project's ICML rule).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Pareto palette (must match scripts/replot_pareto_actual_comparison.py)
COLOR_DISCO   = "black"
COLOR_S1_K1   = "#1f77b4"
COLOR_S1_K2   = "#ff7f0e"
COLOR_INC_QCD = "0.55"   # mid-grey, distinct from black DisCo-only


def _read(csv_path: Path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            r["bin_center"] = float(r["bin_center"])
            r["density"]    = float(r["density"])
            r["n_jets"]     = int(r["n_jets"])
            rows.append(r)
    return rows


def _series(rows, method_prefix):
    sub = [r for r in rows if r["method"].startswith(method_prefix)]
    if not sub:
        return None, None, None, None
    sub.sort(key=lambda r: r["bin_center"])
    cx  = np.array([r["bin_center"] for r in sub])
    h   = np.array([r["density"]    for r in sub])
    n   = sub[0]["n_jets"]
    lam = sub[0]["method"].split("_lambda", 1)[-1]
    return cx, h, n, lam


def _plot_one(csv_path: Path, out_dir: Path, *, s1_k: int, target_eff: float,
              stem: str) -> None:
    rows = _read(csv_path)

    cx_inc, h_inc, _, _ = _series(rows, "inclusive_qcd")
    cx_d,   h_d,   _, lam_d = _series(rows, "disco_only_lambda")
    s1_prefix = f"s1_disco_k{s1_k}_lambda"
    cx_s,   h_s,   _, lam_s = _series(rows, s1_prefix)
    if any(x is None for x in (h_inc, h_d, h_s)):
        raise SystemExit(f"missing series in {csv_path} (k={s1_k})")

    # Recover effective eff from row-level targets is not in the CSV; the
    # original script encoded it into the figure legend, so we re-derive
    # by simply quoting the user-known target_eff for that file. (Keeps
    # this replot script CSV-only and offline.)
    label_inc = "inclusive QCD"
    label_d   = rf"DisCo-only ($\lambda$={lam_d}, eff$\approx${target_eff:.4f})"
    label_s   = rf"S1+DisCo k={s1_k} ($\lambda$={lam_s}, eff$\approx${target_eff:.4f})"

    color_s = COLOR_S1_K1 if s1_k == 1 else COLOR_S1_K2

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.step(cx_inc, h_inc, where="mid", color=COLOR_INC_QCD, lw=1.6, label=label_inc)
    ax.step(cx_d,   h_d,   where="mid", color=COLOR_DISCO,   lw=1.6, label=label_d)
    ax.step(cx_s,   h_s,   where="mid", color=color_s,       lw=1.6, label=label_s)

    ax.set_xlabel(r"jet $m_{SD}$ [GeV]")
    ax.set_ylabel("normalized density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {pdf}")
    print(f"saved {png}")


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=here / "results")
    args = ap.parse_args()

    # k=2 figure (default eff target 0.9475 in the original script)
    _plot_one(
        args.results_dir / "qcd_softdrop_histograms_eff_match.csv",
        args.results_dir,
        s1_k=2, target_eff=0.9475,
        stem="qcd_softdrop_histograms_eff_match",
    )

    # k=1 figure (eff target 0.932 — encoded in the original filename)
    _plot_one(
        args.results_dir / "qcd_softdrop_histograms_eff_match_eff932_k1.csv",
        args.results_dir,
        s1_k=1, target_eff=0.9320,
        stem="qcd_softdrop_histograms_eff_match_eff932_k1",
    )


if __name__ == "__main__":
    main()
