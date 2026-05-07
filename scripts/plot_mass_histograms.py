#!/usr/bin/env python3
"""Compare QCD mass sculpting and Hbb mass shape between two models at lambda=0.

Produces a 2×2 figure:
  Top row   – QCD mass histograms at several Hbb-efficiency cut thresholds
              (inclusive, 50%, 10%, 1%). A sculpting-free model has all cuts
              matching the inclusive distribution.
  Bottom row – Hbb mass histograms (all Hbb jets + jets passing 50% eff cut).
              The Higgs peak should be preserved under S1 projection.

Usage:
  python scripts/plot_mass_histograms.py [--out runs/mass_histograms.png]
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch

plt.style.use(hep.style.CMS)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.experiments.finetune import _load_svd
from src.core.model import PerpClassifier
from src.core.projections import project_svd
from src.core.data import load_h5_group, stratified_split

H5      = "/scope-vol/hidden_out/train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5"
SVD_S1  = "/scope-vol/svd_results/QCD_massshift_cls_tokens_ln_svd.npz"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
BATCH   = 4096
MASS_LO, MASS_HI, NBINS = 75.0, 175.0, 10


def load_val():
    x_all, y_all = load_h5_group(H5, "", "cls_tokens_ln", "label")
    import h5py
    with h5py.File(H5) as f:
        mass_all = f["jet_sdmass"][:]
    # Use full dataset for sculpting plots — inference only, no gradient updates
    return x_all, y_all, mass_all


def infer(model, x, svd_f, svd_fm, svd_r, svd_rm):
    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(x), BATCH):
            xb = torch.from_numpy(x[s:s+BATCH]).to(DEVICE)
            if svd_f is not None:
                xb = project_svd(xb, svd_f, svd_fm, svd_r, svd_rm)
            out = model(xb)
            if isinstance(out, (tuple, list)):
                out = out[0]
            probs.append(torch.softmax(out, dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def load_model(ckpt_path, in_dim, num_classes):
    m = PerpClassifier(in_dim=in_dim, num_classes=num_classes,
                       for_inference=False, use_pretrained_norm_fc=False,
                       input_is_post_ln=True).to(DEVICE)
    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return m


def eff_threshold(hbb_scores, target_eff):
    """Score threshold giving Hbb efficiency ≥ target_eff."""
    return float(np.quantile(hbb_scores, 1.0 - target_eff))


def norm_hist(mass, mask, bins, lo, hi):
    h, _ = np.histogram(mass[mask], bins=bins, range=(lo, hi))
    h = h.astype(float)
    return h / h.sum() if h.sum() > 0 else h


def plot_qcd_sculpting(ax, mass, qcd_mask, hbb_score_qcd, label, color_map):
    effs   = [1.0, 0.5, 0.10, 0.01]
    styles = ["-", "--", "-.", ":"]
    labels = ["Inclusive QCD", "50% Hbb eff cut", "10% Hbb eff cut", "1% Hbb eff cut"]
    bins   = np.linspace(MASS_LO, MASS_HI, NBINS + 1)
    cx     = 0.5 * (bins[:-1] + bins[1:])

    # thresholds are computed on QCD-only scores (simulating "what QCD passes")
    for eff, sty, lab, col in zip(effs, styles, labels, color_map):
        if eff == 1.0:
            mask = qcd_mask
        else:
            thr  = eff_threshold(hbb_score_qcd, eff)   # eff on QCD = mis-id rate
            # here we use efficiency on the Hbb population; threshold from all QCD
            mask = qcd_mask & (hbb_score_qcd[qcd_mask] > thr)[np.cumsum(qcd_mask)-1]
            # simpler: cut on per-sample score
        h = norm_hist(mass, qcd_mask if eff == 1.0 else
                      np.zeros(len(mass), bool), bins, MASS_LO, MASS_HI)
        # recompute properly
        if eff == 1.0:
            sel = qcd_mask
        else:
            thr = np.quantile(hbb_score_qcd, 1.0 - eff)
            sel = qcd_mask & (hbb_score_qcd > thr)
        h, _ = np.histogram(mass[sel], bins=bins, range=(MASS_LO, MASS_HI))
        h = h.astype(float) / max(h.sum(), 1)
        ax.plot(cx, h, sty, color=col, linewidth=1.8, label=lab)

    ax.set_title(label)
    ax.set_xlabel("jet $m_{SD}$ [GeV]")
    ax.set_ylabel("Normalised counts")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_hbb_shape(ax, mass, hbb_mask, hbb_scores_all, models_info):
    bins = np.linspace(MASS_LO, MASS_HI, NBINS + 1)
    cx   = 0.5 * (bins[:-1] + bins[1:])

    for (run_label, color, scores) in models_info:
        # All Hbb jets
        h_all = norm_hist(mass, hbb_mask, bins, MASS_LO, MASS_HI)
        ax.plot(cx, h_all, "-",  color=color, linewidth=2.0, label=f"{run_label} (all Hbb)")

        # Hbb jets passing 50% eff cut
        thr = eff_threshold(scores[hbb_mask], 0.50)
        sel = hbb_mask & (scores > thr)
        h_cut = norm_hist(mass, sel, bins, MASS_LO, MASS_HI)
        ax.plot(cx, h_cut, "--", color=color, linewidth=1.5, label=f"{run_label} (50% eff cut)", alpha=0.8)

    ax.axvline(125.09, color="gray", linestyle=":", linewidth=1.2, label="$m_H$ = 125 GeV")
    ax.set_title("Hbb mass shape")
    ax.set_xlabel("jet $m_{SD}$ [GeV]")
    ax.set_ylabel("Normalised counts")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "runs/mass_histograms.png"))
    args = ap.parse_args()

    print("Loading val data …")
    x_va, y_va, mass_va = load_val()
    qcd_mask = (y_va == 0)
    hbb_mask = (y_va == 1)
    in_dim   = x_va.shape[1]
    n_cls    = 2

    # ── Model A: DisCo-only λ=0 ──────────────────────────────────────────────
    ckpt_a = os.path.join(ROOT, "runs/disco_pareto_pretrained/lambda_0/best.pt")
    print(f"Inference: DisCo-only λ=0 …")
    model_a = load_model(ckpt_a, in_dim, n_cls)
    probs_a = infer(model_a, x_va, None, None, None, None)
    score_a = probs_a[:, 1]   # Hbb probability

    # ── Model B: S1+DisCo k=2 λ=0 ────────────────────────────────────────────
    ckpt_b = os.path.join(ROOT, "runs/cure_disco_pareto__S1_massshift_ep40_k2/lambda_0.0/best.pt")
    svd_f, svd_fm, svd_r, svd_rm = _load_svd(SVD_S1, None, 2, 0.95, DEVICE)
    print(f"Inference: S1+DisCo k=2 λ=0 …")
    model_b = load_model(ckpt_b, in_dim, n_cls)
    probs_b = infer(model_b, x_va, svd_f, svd_fm, svd_r, svd_rm)
    score_b = probs_b[:, 1]

    # ── Plot ──────────────────────────────────────────────────────────────────
    # 3 panels: one per mis-id working point; each overlays inclusive + Vanilla ParT + S1
    misid_rates  = [0.50, 0.10, 0.01]
    panel_titles = ["50% QCD mis-id", "10% QCD mis-id", "1% QCD mis-id"]
    colors  = ["#333333", "#1f77b4", "#d62728"]
    lstyles = ["-", "--", "-."]
    labels  = ["Inclusive QCD", "Vanilla ParT (DisCo λ=0)", "S1+DisCo k=2 (λ=0)"]

    bins = np.linspace(MASS_LO, MASS_HI, NBINS + 1)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax in axes:
        hep.cms.label("Simulation", data=False, ax=ax)

    for ax, rate, title in zip(axes, misid_rates, panel_titles):
        thr_a = np.quantile(score_a[qcd_mask], 1.0 - rate)
        thr_b = np.quantile(score_b[qcd_mask], 1.0 - rate)

        sels = [
            qcd_mask,
            qcd_mask & (score_a > thr_a),
            qcd_mask & (score_b > thr_b),
        ]

        for sel, col, lab, sty in zip(sels, colors, labels, lstyles):
            h, _ = np.histogram(mass_va[sel], bins=bins, range=(MASS_LO, MASS_HI))
            h = h.astype(float) / max(h.sum(), 1)
            n = int(sel.sum())
            hep.histplot(h, bins, ax=ax, histtype="step", linestyle=sty,
                         color=col, linewidth=2, label=f"{lab} (n={n:,})")

        ax.set_title(title)
        ax.set_xlabel(r"jet $m_\mathrm{SD}$ [GeV]")
        ax.set_ylabel("Normalised counts")
        ax.legend(fontsize=10)
        ax.set_xlim(MASS_LO, MASS_HI)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.savefig(args.out.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
