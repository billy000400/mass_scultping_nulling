#!/usr/bin/env python3
"""QCD jet_sdmass histograms at 1% mis-id WP, matched eff between methods.

For each of {inclusive QCD, DisCo-only frontier@eff~target, S1+DisCo k=2
frontier@eff~target}, builds the cut-passing QCD mass histogram and saves
both a CSV (long form) and a comparison plot under results/.

The two trained models picked are the **frontier** runs whose eff_at_1pct
is closest to ``--target-eff`` (default 0.9475). Frontier = upper-envelope
of summary.csv. The check that pareto_actual_comparison.pdf is "real"
is whether S1+DisCo k=2 has a visibly less-sculpted mass spectrum than
DisCo-only at matched eff.

Usage:
    cd /scope-vol/mass-sculpting-nulling
    python scripts/qcd_softdrop_histograms_eff_match.py
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from src.core.data import load_h5_group, load_aux_from_h5, stratified_split  # noqa: E402
from src.core.model import PerpClassifier  # noqa: E402
from src.core.projections import load_svd_basis  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("qcd_softdrop_eff_match")

DEFAULT_H5 = "/scope-vol/hidden_out/train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5"
S1_BASIS = "/scope-vol/svd_results/QCD_massshift_cls_tokens_ln_svd.npz"


def frontier_closest(summary_csv: Path, target_eff: float, jsd_lo: float = 0.005, jsd_hi: float = 0.06):
    """Find the frontier point of summary.csv closest in eff_at_1pct to `target_eff`."""
    rows = list(csv.DictReader(open(summary_csv)))
    rows = [r for r in rows if jsd_lo <= float(r["jsd"]) <= jsd_hi]
    sorted_by_jsd = sorted(
        ((float(r["jsd"]), float(r["eff_at_1pct"]), r["disco_lambda"]) for r in rows),
        key=lambda x: x[0],
    )
    keep, m = [], -np.inf
    for jj, ee, lam in sorted_by_jsd:
        if ee > m:
            keep.append((jj, ee, lam))
            m = ee
    return min(keep, key=lambda x: abs(x[1] - target_eff))


def load_val(h5: str, *, seed: int = 1337, train_frac: float = 0.8):
    x_all, y_all = load_h5_group(h5, "", "cls_tokens_ln", "label", data_fraction=1.0)
    if y_all is None:
        raise RuntimeError("No labels in H5")
    train_idx, val_idx = stratified_split(y_all, train_frac, seed)
    aux_all = load_aux_from_h5(h5, "", "jet_sdmass", len(x_all))
    return x_all[val_idx], y_all[val_idx], aux_all[val_idx]


def project_features(x: np.ndarray, basis_path: Optional[str], k: Optional[int]) -> np.ndarray:
    if basis_path is None:
        return x
    Vh, mean, k_used = load_svd_basis(basis_path, k, 0.95)
    logger.info("SVD forget: %s, k_used=%d", os.path.basename(basis_path), k_used)
    Vh_t = torch.from_numpy(Vh.astype(np.float32))
    mean_t = torch.from_numpy(mean.astype(np.float32))
    x_t = torch.from_numpy(x.astype(np.float32))
    return (x_t - ((x_t - mean_t) @ Vh_t.T) @ Vh_t).numpy()


def forward_probs(state_dict_path: str, x: np.ndarray, *, device: str, batch_size: int = 16384) -> np.ndarray:
    sd = torch.load(state_dict_path, map_location=device)
    if isinstance(sd, dict) and "model_state" in sd:
        sd = sd["model_state"]
    model = PerpClassifier(
        in_dim=x.shape[1], num_classes=2, for_inference=False,
        use_bias=True, input_is_post_ln=True,
    )
    model.load_state_dict(sd)
    model.to(device).eval()
    out = np.empty((len(x), 2), dtype=np.float32)
    x_t = torch.from_numpy(x.astype(np.float32))
    with torch.no_grad():
        for i in range(0, len(x_t), batch_size):
            xb = x_t[i:i + batch_size].to(device)
            logits = model(xb)
            out[i:i + batch_size] = F.softmax(logits, dim=1).cpu().numpy()
    return out


def cut_passing_qcd_mass(probs: np.ndarray, targets: np.ndarray, mass: np.ndarray,
                         *, target_misid_rate: float = 0.01) -> tuple[np.ndarray, float, int]:
    """Return (mass values of QCD jets passing the TXbb cut at given mis-id, threshold, n_qcd_total)."""
    qcd_mask = targets == 0
    txbb = probs[:, 1] / (probs[:, 1] + probs[:, 0] + 1e-12)
    threshold = float(np.percentile(txbb[qcd_mask], (1.0 - target_misid_rate) * 100.0))
    pass_mask = qcd_mask & (txbb > threshold)
    return mass[pass_mask], threshold, int(qcd_mask.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--target-eff", type=float, default=0.9475)
    ap.add_argument("--target-misid", type=float, default=0.01)
    ap.add_argument("--n-bins", type=int, default=50)
    ap.add_argument("--mass-min", type=float, default=None)
    ap.add_argument("--mass-max", type=float, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-dir", type=Path, default=_PROJ / "results")
    ap.add_argument("--stem", default="qcd_softdrop_histograms_eff_match")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Identify the two models to compare.
    disco_only = frontier_closest(_PROJ / "runs/disco_pareto_pretrained/summary.csv", args.target_eff)
    s1_k2      = frontier_closest(_PROJ / "runs/cure_disco_pareto__S1_massshift_ep40_k2/summary.csv", args.target_eff)
    print()
    print(f"  Target Eff@{int(round(args.target_misid*100))}% = {args.target_eff}")
    print(f"  DisCo-only      → λ={disco_only[2]}  eff={disco_only[1]:.4f}  jsd_at_cut={disco_only[0]:.4f}")
    print(f"  S1+DisCo k=2    → λ={s1_k2[2]}      eff={s1_k2[1]:.4f}  jsd_at_cut={s1_k2[0]:.4f}")
    print()

    # Load val features once.
    x_va, y_va, mass_va = load_val(args.h5)
    logger.info("val: x %s, mass %s, n_qcd=%d, n_hbb=%d",
                x_va.shape, mass_va.shape, int((y_va == 0).sum()), int((y_va == 1).sum()))

    # Histogram axis based on inclusive QCD mass range (matches earlier convention).
    mass_qcd = mass_va[y_va == 0]
    lo = float(args.mass_min) if args.mass_min is not None else float(mass_qcd.min())
    hi = float(args.mass_max) if args.mass_max is not None else float(mass_qcd.max())
    bins = np.linspace(lo, hi, args.n_bins + 1)
    bin_lo = bins[:-1]
    bin_hi = bins[1:]
    bin_center = 0.5 * (bin_lo + bin_hi)
    bin_width = bins[1] - bins[0]

    def density_hist(x):
        h, _ = np.histogram(x, bins=bins)
        n = h.sum()
        return h.astype(np.float64) / (n * bin_width) if n > 0 else h.astype(np.float64), int(n)

    rows: list[dict] = []

    # Inclusive QCD
    h_inc, n_inc = density_hist(mass_qcd)
    for i in range(len(bin_center)):
        rows.append({
            "working_point": f"{int(round(args.target_misid*100))}% QCD mis-id",
            "method": "inclusive_qcd",
            "bin_lo": bin_lo[i], "bin_hi": bin_hi[i], "bin_center": bin_center[i],
            "density": h_inc[i], "n_jets": n_inc,
        })

    # Run inference for each model and compute cut-passing QCD mass histogram
    def _lambda_dir(parent: Path, lam_str: str) -> Path:
        """Different sweeps used different conventions: 'lambda_20' vs 'lambda_20.0'.
        Try the literal form first, then the int-stripped form."""
        for cand in (parent / f"lambda_{lam_str}",
                     parent / f"lambda_{lam_str.rstrip('0').rstrip('.')}",
                     parent / f"lambda_{int(float(lam_str))}"):
            if cand.exists():
                return cand
        raise FileNotFoundError(f"No lambda dir for lam={lam_str!r} under {parent}")

    def _lam_label(prefix: str, lam_str: str) -> str:
        # Sanitize "20.0" -> "20", "0.5" -> "0p5" for filename-friendly labels.
        s = lam_str.rstrip("0").rstrip(".") if "." in lam_str else lam_str
        return f"{prefix}_lambda{s.replace('.', 'p')}"

    runs = [
        {
            "label": _lam_label("disco_only", disco_only[2]),
            "best_pt": _lambda_dir(_PROJ / "runs/disco_pareto_pretrained", disco_only[2]) / "best.pt",
            "basis": None, "k": None, "lam": disco_only[2],
        },
        {
            "label": _lam_label("s1_disco_k2", s1_k2[2]),
            "best_pt": _lambda_dir(_PROJ / "runs/cure_disco_pareto__S1_massshift_ep40_k2", s1_k2[2]) / "best.pt",
            "basis": S1_BASIS, "k": 2, "lam": s1_k2[2],
        },
    ]

    for r in runs:
        logger.info("=== %s (best_pt=%s) ===", r["label"], r["best_pt"])
        x_proj = project_features(x_va, r["basis"], r["k"])
        probs = forward_probs(str(r["best_pt"]), x_proj, device=args.device)
        masses_pass, thr, n_qcd_total = cut_passing_qcd_mass(
            probs, y_va, mass_va, target_misid_rate=args.target_misid,
        )
        logger.info("  TXbb thr = %.6f, cut-passing QCD = %d (≈ %.3f%% of %d)",
                    thr, len(masses_pass), 100.0 * len(masses_pass) / n_qcd_total, n_qcd_total)
        h_pass, n_pass = density_hist(masses_pass)
        for i in range(len(bin_center)):
            rows.append({
                "working_point": f"{int(round(args.target_misid*100))}% QCD mis-id",
                "method": r["label"],
                "bin_lo": bin_lo[i], "bin_hi": bin_hi[i], "bin_center": bin_center[i],
                "density": h_pass[i], "n_jets": n_pass,
            })

    csv_path = args.out_dir / f"{args.stem}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "working_point", "method", "bin_lo", "bin_hi", "bin_center", "density", "n_jets",
        ])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    logger.info("wrote %s", csv_path)

    # Plot
    import matplotlib.pyplot as plt
    methods_to_plot = [
        ("inclusive_qcd",         "inclusive QCD",             "black", "-"),
        (runs[0]["label"], f"DisCo-only (λ={disco_only[2]}, eff={disco_only[1]:.4f})", "#1f77b4", "-"),
        (runs[1]["label"], f"S1+DisCo k=2 (λ={s1_k2[2]}, eff={s1_k2[1]:.4f})",       "#d62728", "-"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for method, label, color, ls in methods_to_plot:
        sub = [r for r in rows if r["method"] == method]
        sub.sort(key=lambda r: r["bin_center"])
        x = np.array([r["bin_center"] for r in sub])
        y = np.array([r["density"]    for r in sub])
        ax.step(x, y, where="mid", color=color, linestyle=ls, lw=1.6, label=label)
    ax.set_xlabel("jet $m_{SD}$ [GeV]")
    ax.set_ylabel("normalized density")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    pdf = args.out_dir / f"{args.stem}.pdf"
    png = args.out_dir / f"{args.stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s and %s", pdf, png)


if __name__ == "__main__":
    main()
