#!/usr/bin/env python3
"""Compare Hbb signal sdmass shape at matched working points for two models.

Usage:
  python scripts/plot_signal_shape.py \
      --h5 /path/to/hidden_states.h5 \
      --model-a runs/disco_pareto_pretrained/lambda_20/best.pt \
      --label-a "DisCo-only λ=20" \
      --model-b runs/cure_disco_pareto__S1_massshift_ep40_k1/lambda_10.0/best.pt \
      --label-b "S1 k=1 (40ep) λ=10" \
      --svd-forget-b /scope-vol/svd_results/QCD_massshift_cls_tokens_ln_svd.npz \
      --svd-k-b 1 \
      --out runs/signal_shape_comparison.png
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Minimal model
# ---------------------------------------------------------------------------

class Head(nn.Module):
    def __init__(self, in_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # input is already post-LN; norm weights kept but not applied


def load_head(ckpt_path: str) -> Head:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    m = Head()
    m.load_state_dict(state)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# SVD projection
# ---------------------------------------------------------------------------

def load_svd_forget(npz_path: str, k: int):
    d = np.load(npz_path)
    Vh = d["Vh"][:k].astype(np.float32)
    mean = d["mean"].astype(np.float32)
    return Vh, mean


def apply_svd_forget(x: np.ndarray, Vh: np.ndarray, mean: np.ndarray) -> np.ndarray:
    x_c = x - mean
    proj = (x_c @ Vh.T) @ Vh
    return x - proj


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_val_data(h5_path: str, train_frac: float = 0.8, seed: int = 1337):
    with h5py.File(h5_path, "r") as f:
        x = np.array(f["cls_tokens_ln"]).reshape(len(f["cls_tokens_ln"]), -1)
        y = np.array(f["label"]).reshape(-1)
        mass = np.array(f["jet_sdmass"]).reshape(-1)

    y = np.rint(y).astype(int)
    rng = np.random.RandomState(seed)
    val_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = int(round(len(idx) * train_frac))
        val_idx.append(idx[k:])
    val_idx = np.concatenate(val_idx)

    return x[val_idx], y[val_idx], mass[val_idx]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def get_scores(x: np.ndarray, model: Head, batch_size: int = 8192) -> np.ndarray:
    """Returns softmax prob for class 1 (Hbb)."""
    scores = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i:i + batch_size])
            logits = model(xb)
            prob = F.softmax(logits, dim=1)[:, 1]
            scores.append(prob.numpy())
    return np.concatenate(scores)


def threshold_at_fpr(scores_bg: np.ndarray, fpr_target: float = 0.01) -> float:
    return float(np.quantile(scores_bg, 1.0 - fpr_target))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_signal_shapes(
    results: list[dict],
    out_path: str,
    bins: np.ndarray,
    fpr_target: float = 0.01,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    # Left: mass shapes overlaid
    ax = axes[0]
    for i, r in enumerate(results):
        mass_sig = r["mass_sig_pass"]
        w = np.ones(len(mass_sig)) / len(mass_sig)
        ax.hist(mass_sig, bins=bins, weights=w, histtype="step",
                linewidth=2, color=colors[i], label=r["label"])
    ax.set_xlabel("Soft-drop mass [GeV]")
    ax.set_ylabel("Normalised fraction")
    ax.set_title(f"Hbb signal shape after {fpr_target*100:.0f}% QCD WP cut")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: ratio of shapes (B / A)
    ax = axes[1]
    ref_h, _ = np.histogram(results[0]["mass_sig_pass"], bins=bins,
                             density=True)
    for i, r in enumerate(results[1:], start=1):
        h, _ = np.histogram(r["mass_sig_pass"], bins=bins, density=True)
        ratio = np.where(ref_h > 0, h / np.where(ref_h > 0, ref_h, 1), np.nan)
        centres = 0.5 * (bins[:-1] + bins[1:])
        ax.step(centres, ratio, where="mid", linewidth=2, color=colors[i],
                label=f"{r['label']} / {results[0]['label']}")
    ax.axhline(1, color=colors[0], linestyle="--", linewidth=1, alpha=0.7,
               label=results[0]["label"] + " (reference)")
    ax.set_xlabel("Soft-drop mass [GeV]")
    ax.set_ylabel("Ratio to reference")
    ax.set_title("Shape ratio")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotate n_pass and threshold
    for i, r in enumerate(results):
        label_str = (f"{r['label']}\n"
                     f"  threshold={r['threshold']:.4f}\n"
                     f"  N_sig_pass={r['n_sig_pass']}\n"
                     f"  eff@{fpr_target*100:.0f}%={r['eff']:.4f}")
        axes[0].text(0.02, 0.97 - i * 0.18, label_str, transform=axes[0].transAxes,
                     fontsize=7.5, verticalalignment="top",
                     color=colors[i], bbox=dict(boxstyle="round,pad=0.2",
                                                 facecolor="white", alpha=0.7))

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.rsplit(".", 1)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--label-a", default="Model A")
    ap.add_argument("--svd-forget-a", default=None)
    ap.add_argument("--svd-k-a", type=int, default=None)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--label-b", default="Model B")
    ap.add_argument("--svd-forget-b", default=None)
    ap.add_argument("--svd-k-b", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--mass-bins", type=int, default=40)
    args = ap.parse_args()

    print("Loading val data...")
    x, y, mass = load_val_data(args.h5, args.train_frac, args.seed)
    sig_mask = y == 1
    bg_mask = y == 0
    print(f"  Val set: {sig_mask.sum()} Hbb, {bg_mask.sum()} QCD")

    bins = np.linspace(float(mass.min()), float(mass.max()), args.mass_bins + 1)

    configs = [
        (args.model_a, args.label_a, args.svd_forget_a, args.svd_k_a),
        (args.model_b, args.label_b, args.svd_forget_b, args.svd_k_b),
    ]

    results = []
    for ckpt, label, svd_path, svd_k in configs:
        print(f"\nInference: {label}")
        x_proj = x.copy()
        if svd_path is not None:
            print(f"  Applying SVD forget: {svd_path} k={svd_k}")
            Vh, mean = load_svd_forget(svd_path, svd_k)
            x_proj = apply_svd_forget(x_proj, Vh, mean)

        model = load_head(ckpt)
        scores = get_scores(x_proj, model)

        thresh = threshold_at_fpr(scores[bg_mask], args.fpr)
        pass_mask = scores > thresh
        sig_pass = sig_mask & pass_mask
        eff = sig_pass.sum() / sig_mask.sum()
        actual_fpr = (bg_mask & pass_mask).sum() / bg_mask.sum()

        print(f"  threshold={thresh:.4f}  eff={eff:.4f}  actual_fpr={actual_fpr:.4f}  N_sig_pass={sig_pass.sum()}")
        results.append({
            "label": label,
            "threshold": thresh,
            "eff": float(eff),
            "n_sig_pass": int(sig_pass.sum()),
            "mass_sig_pass": mass[sig_pass],
        })

    plot_signal_shapes(results, args.out, bins, args.fpr)


if __name__ == "__main__":
    main()
