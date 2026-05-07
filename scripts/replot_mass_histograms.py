#!/usr/bin/env python3
"""Replot the QCD soft-drop-mass histograms from a previously dumped CSV.

The companion script `plot_mass_histograms.py` runs inference on the
JetClass H5 + checkpoints (heavy) and now also writes a long-format
CSV (`results/qcd_softdrop_histograms.csv`) of bin densities. This
replot script reads that CSV and rebuilds the publication figure
locally, so layout / font / annotation tweaks do not require rerunning
the GPU job.

Usage:
  python scripts/replot_mass_histograms.py
  python scripts/replot_mass_histograms.py --csv path/to/file.csv \
                                            --out results/qcd_softdrop_histograms.pdf \
                                            --annotate-jsd

The CSV is in long format with columns:
  working_point, method, bin_lo, bin_hi, bin_center, density, n_jets
where working_point is one of {"inclusive", "50% QCD mis-id",
"10% QCD mis-id", "1% QCD mis-id"} and method is one of
{"inclusive_qcd", "vanilla_part_lambda0", "s1_k2_alpha1_lambda0"}.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PANEL_ORDER = ["50% QCD mis-id", "10% QCD mis-id", "1% QCD mis-id"]


def _read_csv(csv_path: Path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            r["bin_lo"] = float(r["bin_lo"])
            r["bin_hi"] = float(r["bin_hi"])
            r["bin_center"] = float(r["bin_center"])
            r["density"] = float(r["density"])
            r["n_jets"] = int(r["n_jets"])
            rows.append(r)
    return rows


def _series(rows, working_point, method):
    sub = [r for r in rows if r["working_point"] == working_point
                              and r["method"] == method]
    if not sub:
        return None, None, None
    sub = sorted(sub, key=lambda r: r["bin_lo"])
    cx = np.array([r["bin_center"] for r in sub])
    dens = np.array([r["density"] for r in sub])
    n = sub[0]["n_jets"]
    return cx, dens, n


def _bin_edges(rows):
    sub = sorted({(r["bin_lo"], r["bin_hi"]) for r in rows})
    edges = [lo for lo, _ in sub] + [sub[-1][1]]
    return np.array(edges)


def _jsd(p, q, eps=1e-10):
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _ks(p, q):
    """1D Kolmogorov--Smirnov distance between two normalised
    histograms with identical bin edges (max |CDF difference|)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    return float(np.max(np.abs(np.cumsum(p) - np.cumsum(q))))


def _chi2_per_ndf(n_sel, n_inc, p_inc, eps=1e-10):
    """Pearson chi^2 per NDF between observed selected-QCD counts and
    expected counts inferred from the inclusive (binomial) shape.

    n_sel: array of selected-QCD bin counts (integers)
    n_inc: total selected-QCD count (n_sel.sum())
    p_inc: inclusive QCD bin densities (sum to 1)
    """
    n_sel = np.asarray(n_sel, dtype=float)
    expected = n_inc * np.asarray(p_inc, dtype=float)
    nz = expected > eps
    chi2 = np.sum((n_sel[nz] - expected[nz]) ** 2 / expected[nz])
    ndf = max(int(nz.sum()) - 1, 1)
    return float(chi2 / ndf), int(ndf)


def _bootstrap_jsd_ks(n_sel_total, p_sel, p_ref, n_boot=500, seed=0):
    """Resample bin counts of the selected-QCD histogram (multinomial
    with cell probabilities p_sel and total n_sel_total) and recompute
    JSD/KS against p_ref each draw.  Returns (mean, std) of each."""
    rng = np.random.default_rng(seed)
    p = np.asarray(p_sel, dtype=float)
    p = p / max(p.sum(), 1e-12)
    jsd_vals, ks_vals = [], []
    for _ in range(n_boot):
        counts = rng.multinomial(n_sel_total, p)
        h = counts.astype(float) / max(counts.sum(), 1)
        jsd_vals.append(_jsd(h, p_ref))
        ks_vals.append(_ks(h, p_ref))
    return (
        float(np.mean(jsd_vals)), float(np.std(jsd_vals)),
        float(np.mean(ks_vals)),  float(np.std(ks_vals)),
    )


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path,
                    default=here / "results" / "qcd_softdrop_histograms.csv")
    ap.add_argument("--out", type=Path,
                    default=here / "results" / "qcd_softdrop_histograms.pdf")
    ap.add_argument("--annotate-jsd", action="store_true",
                    help="Print per-panel JSD against inclusive QCD on each subplot.")
    ap.add_argument("--ratio-panel", action="store_true",
                    help="Add a bottom row of selected/inclusive density-ratio "
                         "panels to the figure.")
    ap.add_argument("--metrics-out", type=Path,
                    default=None,
                    help="If set, write a CSV of per-(working_point, method) "
                         "JSD / KS / chi^2 / bootstrap CIs to this path.")
    ap.add_argument("--n-bootstrap", type=int, default=500,
                    help="Bootstrap draws for JSD/KS uncertainty (default 500).")
    args = ap.parse_args()

    rows = _read_csv(args.csv)
    edges = _bin_edges(rows)
    mass_lo, mass_hi = float(edges[0]), float(edges[-1])
    print(f"loaded {len(rows)} rows, bins=[{mass_lo:.0f}, {mass_hi:.0f}] GeV"
          f" ({len(edges)-1} bins)")

    cx_inc, h_inc, n_inc = _series(rows, "inclusive", "inclusive_qcd")
    if cx_inc is None:
        raise SystemExit("inclusive QCD reference not found in CSV")

    label_inc = f"Inclusive QCD"
    label_a   = r"Vanilla ParT  ($\lambda_\mathrm{DisCo}=0$)"
    label_b   = r"S1 edit, $k{=}2$  ($\lambda_\mathrm{DisCo}=0$)"

    if args.ratio_panel:
        fig, axes_grid = plt.subplots(
            2, 3, figsize=(7.0, 3.0), sharex="col",
            gridspec_kw={"height_ratios": [3, 1]},
        )
        top_axes = axes_grid[0]
        bot_axes = axes_grid[1]
    else:
        fig, top_axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
        bot_axes = [None, None, None]

    eps_inc = np.where(h_inc > 1e-10, h_inc, 1e-10)

    for top_ax, bot_ax, title in zip(top_axes, bot_axes, PANEL_ORDER):
        cx_a, h_a, n_a = _series(rows, title, "vanilla_part_lambda0")
        cx_b, h_b, n_b = _series(rows, title, "s1_k2_alpha1_lambda0")
        if h_a is None or h_b is None:
            raise SystemExit(f"missing data for working point {title!r}")

        top_ax.step(cx_inc, h_inc, where="mid", color="black",
                    linestyle="-",  linewidth=1.3, label=label_inc)
        top_ax.step(cx_a,   h_a,   where="mid", color="#1f77b4",
                    linestyle="--", linewidth=1.3, label=label_a)
        top_ax.step(cx_b,   h_b,   where="mid", color="#d62728",
                    linestyle="-.", linewidth=1.3, label=label_b)

        top_ax.set_title(title, fontsize=8)
        top_ax.tick_params(labelsize=7)
        top_ax.grid(alpha=0.25)
        top_ax.set_xlim(mass_lo, mass_hi)

        if args.annotate_jsd:
            jsd_a = _jsd(h_a, h_inc)
            jsd_b = _jsd(h_b, h_inc)
            top_ax.text(0.05, 0.95,
                        f"JSD$_\mathrm{{vanilla}}$={jsd_a:.3f}\n"
                        f"JSD$_\mathrm{{S1}}$={jsd_b:.3f}",
                        transform=top_ax.transAxes, ha="left", va="top",
                        fontsize=6,
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.7, pad=1.0))

        if bot_ax is not None:
            ratio_a = h_a / eps_inc
            ratio_b = h_b / eps_inc
            bot_ax.step(cx_a, ratio_a, where="mid",
                        color="#1f77b4", linestyle="--", linewidth=1.2)
            bot_ax.step(cx_b, ratio_b, where="mid",
                        color="#d62728", linestyle="-.", linewidth=1.2)
            bot_ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
            bot_ax.tick_params(labelsize=7)
            bot_ax.grid(alpha=0.25)
            bot_ax.set_xlim(mass_lo, mass_hi)
            bot_ax.set_xlabel(r"jet $m_\mathrm{SD}$ [GeV]", fontsize=8)
        else:
            top_ax.set_xlabel(r"jet $m_\mathrm{SD}$ [GeV]", fontsize=8)

    top_axes[0].set_ylabel("Normalised counts", fontsize=8)
    if args.ratio_panel:
        bot_axes[0].set_ylabel("ratio", fontsize=8)

    handles, labels = top_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    png = args.out.with_suffix(".png")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"saved {args.out}")
    print(f"saved {png}")

    # ----- selected-QCD metrics + bootstrap -----------------------
    metrics_path = (args.metrics_out
                    or args.out.parent / "qcd_softdrop_metrics.csv")
    metric_rows = []
    for title in PANEL_ORDER:
        for method_key in ("vanilla_part_lambda0", "s1_k2_alpha1_lambda0"):
            cx, h_sel, n_sel = _series(rows, title, method_key)
            if h_sel is None:
                continue
            jsd_pt = _jsd(h_sel, h_inc)
            ks_pt  = _ks(h_sel, h_inc)
            chi2_per_ndf, ndf = _chi2_per_ndf(
                n_sel * h_sel,  # bin counts ≈ density × n_jets
                n_inc=n_sel,
                p_inc=h_inc,
            )
            jsd_mean, jsd_std, ks_mean, ks_std = _bootstrap_jsd_ks(
                n_sel_total=n_sel, p_sel=h_sel, p_ref=h_inc,
                n_boot=args.n_bootstrap, seed=hash(title + method_key) & 0xFFFF,
            )
            metric_rows.append({
                "working_point": title, "method": method_key,
                "n_sel": int(n_sel),
                "jsd": jsd_pt, "jsd_boot_mean": jsd_mean,
                "jsd_boot_std": jsd_std,
                "ks": ks_pt, "ks_boot_mean": ks_mean,
                "ks_boot_std": ks_std,
                "chi2_per_ndf": chi2_per_ndf, "ndf": ndf,
            })
            print(f"  {title:18s} {method_key:25s} "
                  f"n={n_sel:7d}  "
                  f"JSD={jsd_pt:.4f}±{jsd_std:.4f}  "
                  f"KS={ks_pt:.4f}±{ks_std:.4f}  "
                  f"chi2/ndf={chi2_per_ndf:.2f} (ndf={ndf})")

    if metric_rows:
        with open(metrics_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
            w.writeheader()
            w.writerows(metric_rows)
        print(f"saved {metrics_path}")


if __name__ == "__main__":
    main()
