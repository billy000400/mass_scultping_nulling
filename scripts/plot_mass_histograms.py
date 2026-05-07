#!/usr/bin/env python3
"""Compare QCD mass sculpting between Vanilla ParT (DisCo lambda=0) and
S1 latent edit + DisCo at lambda=0, both evaluated at three QCD mis-id
working points (50%, 10%, 1%).

Output: a publication-grade three-panel figure tuned for ICML two-column
layout, intended to be embedded as a figure* (full text-width) float.

Usage:
  python scripts/plot_mass_histograms.py [--out results/qcd_softdrop_histograms.pdf]

Compared to the previous version, this script:
  * drops mplhep.style.CMS / hep.cms.label() (no "CMS Simulation" label),
  * uses figsize (7.0, 2.2) tuned for one-row, three-panel full-width,
  * shares the y-axis across panels and emits a single shared legend
    above the panels (no per-panel legend boxes),
  * uses publication-grade font sizes (7--8 pt) and line widths (1.3 pt),
  * shortens panel titles to "50% / 10% / 1% QCD mis-id".

Optional: pass --annotate-jsd to print per-panel JSD against inclusive
QCD on each subplot.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.experiments.finetune import _load_svd
from src.core.model import PerpClassifier
from src.core.projections import project_svd
from src.core.data import load_h5_group  # full set used; no train/val split

H5      = "/scope-vol/hidden_out/train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5"
SVD_S1  = "/scope-vol/svd_results/QCD_massshift_cls_tokens_ln_svd.npz"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
BATCH   = 4096
MASS_LO, MASS_HI, NBINS = 75.0, 175.0, 10


# -- data / inference ---------------------------------------------------

def load_full():
    x, y = load_h5_group(H5, "", "cls_tokens_ln", "label")
    import h5py
    with h5py.File(H5) as f:
        mass = f["jet_sdmass"][:]
    return x, y, mass


def load_model(ckpt_path, in_dim, num_classes):
    m = PerpClassifier(in_dim=in_dim, num_classes=num_classes,
                       for_inference=False, use_pretrained_norm_fc=False,
                       input_is_post_ln=True).to(DEVICE)
    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return m


@torch.no_grad()
def infer(model, x, svd_f, svd_fm, svd_r, svd_rm):
    model.eval()
    probs = []
    for s in range(0, len(x), BATCH):
        xb = torch.from_numpy(x[s:s+BATCH]).to(DEVICE)
        if svd_f is not None:
            xb = project_svd(xb, svd_f, svd_fm, svd_r, svd_rm)
        out = model(xb)
        if isinstance(out, (tuple, list)):
            out = out[0]
        probs.append(torch.softmax(out, dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0)


# -- helpers ------------------------------------------------------------

def _norm_hist(mass_subset, bins):
    h, _ = np.histogram(mass_subset, bins=bins)
    h = h.astype(float)
    return h / max(h.sum(), 1.0)


def _jsd(p, q, eps=1e-10):
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# -- main ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=os.path.join(ROOT, "results/qcd_softdrop_histograms.pdf"))
    ap.add_argument("--annotate-jsd", action="store_true",
                    help="Print per-panel JSD against inclusive QCD on each subplot.")
    args = ap.parse_args()

    print("Loading full data ...")
    x, y, mass = load_full()
    qcd_mask = (y == 0)
    in_dim = x.shape[1]
    n_cls = 2

    # Model A: DisCo-only (Vanilla ParT linear head, lambda=0)
    ckpt_a = os.path.join(
        ROOT, "runs/disco_pareto_pretrained/lambda_0/best.pt")
    print("Inference: Vanilla ParT (DisCo lambda=0) ...")
    model_a = load_model(ckpt_a, in_dim, n_cls)
    probs_a = infer(model_a, x, None, None, None, None)
    score_a = probs_a[:, 1]  # P(Hbb)

    # Model B: S1 + DisCo at k=2, lambda=0 (no DisCo penalty applied)
    ckpt_b = os.path.join(
        ROOT, "runs/cure_disco_pareto__S1_massshift_ep40_k2/lambda_0.0/best.pt")
    svd_f, svd_fm, svd_r, svd_rm = _load_svd(SVD_S1, None, 2, 0.95, DEVICE)
    print("Inference: S1 latent edit, k=2, alpha=1, DisCo lambda=0 ...")
    model_b = load_model(ckpt_b, in_dim, n_cls)
    probs_b = infer(model_b, x, svd_f, svd_fm, svd_r, svd_rm)
    score_b = probs_b[:, 1]

    # Plot
    misid_rates  = [0.50, 0.10, 0.01]
    panel_titles = ["50% QCD mis-id", "10% QCD mis-id", "1% QCD mis-id"]
    bins         = np.linspace(MASS_LO, MASS_HI, NBINS + 1)
    cx           = 0.5 * (bins[:-1] + bins[1:])

    label_inc = "Inclusive QCD"
    label_a   = r"Vanilla ParT  ($\lambda_\mathrm{DisCo}=0$)"
    label_b   = r"S1 edit, $k{=}2$  ($\lambda_\mathrm{DisCo}=0$)"

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)

    # Inclusive QCD reference (same for all three panels)
    h_inc = _norm_hist(mass[qcd_mask], bins)

    for ax, rate, title in zip(axes, misid_rates, panel_titles):
        thr_a = np.quantile(score_a[qcd_mask], 1.0 - rate)
        thr_b = np.quantile(score_b[qcd_mask], 1.0 - rate)
        sel_a = qcd_mask & (score_a > thr_a)
        sel_b = qcd_mask & (score_b > thr_b)

        h_a = _norm_hist(mass[sel_a], bins)
        h_b = _norm_hist(mass[sel_b], bins)

        # mplhep-style step histograms via matplotlib `step`
        ax.step(cx, h_inc, where="mid", color="black",
                linestyle="-",  linewidth=1.3, label=label_inc)
        ax.step(cx, h_a,   where="mid", color="#1f77b4",
                linestyle="--", linewidth=1.3, label=label_a)
        ax.step(cx, h_b,   where="mid", color="#d62728",
                linestyle="-.", linewidth=1.3, label=label_b)

        ax.set_title(title, fontsize=8)
        ax.set_xlabel(r"jet $m_\mathrm{SD}$ [GeV]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        ax.set_xlim(MASS_LO, MASS_HI)

        if args.annotate_jsd:
            jsd_a = _jsd(h_a, h_inc)
            jsd_b = _jsd(h_b, h_inc)
            ax.text(0.05, 0.95,
                    f"JSD$_a$={jsd_a:.3f}\nJSD$_b$={jsd_b:.3f}",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=6,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.7, pad=1.0))

    axes[0].set_ylabel("Normalised counts", fontsize=8)

    # Single shared legend across the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.rsplit(".", 1)[0] + ".png",
                bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
