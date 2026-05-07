#!/usr/bin/env python3
"""Scree plot of singular values of the centred mass-bin mean matrix M.

Computes M directly from the cached HDF5 of CLS embeddings:
  - filter to QCD jets (label == 0),
  - bin them into B equal-occupancy quantile bins of jet_sdmass,
  - compute the per-bin mean and the overall QCD mean,
  - SVD the centred bin-mean matrix.

This replicates the construction in
``src/data_processing/build_mass_forget_bases.py::build_s1_qcd_massshift``,
but reads directly from the H5 so it can be run locally without the
GPU-host /scope-vol paths or the precomputed .npz basis.

Outputs:
  results/scree_M.pdf, .png
  results/scree_values.csv  (sigma_i / sigma_1 per row)

Usage:
  python scripts/replot_scree_M.py
  python scripts/replot_scree_M.py --h5 path/to/file.h5 --n-bins 20
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--h5", type=Path,
        default=here / "latent" /
                "train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5",
    )
    ap.add_argument("--latent-key", default="cls_tokens_ln")
    ap.add_argument("--label-key",  default="label")
    ap.add_argument("--mass-key",   default="jet_sdmass")
    ap.add_argument("--n-bins", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=here / "results")
    ap.add_argument("--stem", default="scree_M")
    args = ap.parse_args()

    print(f"Reading {args.h5} ...")
    with h5py.File(args.h5, "r") as f:
        labels = np.asarray(f[args.label_key][:]).reshape(-1).astype(np.int64)
        qcd_idx = np.where(labels == 0)[0]
        print(f"  total jets={labels.size:,}  QCD jets={qcd_idx.size:,}")
        # h5py fancy indexing requires sorted, contiguous-friendly indices
        Xq = np.asarray(f[args.latent_key][qcd_idx], dtype=np.float64)
        mq = np.asarray(f[args.mass_key][qcd_idx],  dtype=np.float64)

    # H5 may store cls_tokens_ln with shape (N, 1, D) and jet_sdmass as
    # (N, 1, 1); squeeze them to (N, D) and (N,)
    Xq = Xq.reshape(Xq.shape[0], -1)
    mq = mq.reshape(-1)
    print(f"  Xq shape={Xq.shape}  mq shape={mq.shape}")

    # Build M: per-bin mean centred by overall QCD mean
    bins = _quantile_bin(mq, args.n_bins)
    means = np.stack(
        [Xq[bins == b].mean(axis=0) for b in range(args.n_bins)], axis=0
    )
    qcd_mean = Xq.mean(axis=0)
    M = means - qcd_mean
    print(f"  M shape={M.shape}  (B={args.n_bins}, D={M.shape[1]})")

    # SVD
    _, S, _ = np.linalg.svd(M, full_matrices=False)
    if S.size == 0 or S[0] == 0:
        raise SystemExit("M is degenerate")
    ratio = S / S[0]

    # Numerically-zero tail; drop it
    floor = 1e-7
    keep = ratio > floor
    ratio_p = ratio[keep]
    idx = np.arange(1, ratio_p.size + 1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "scree_values.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "sigma_i", "sigma_i_over_sigma_1"])
        for i, (s_i, r_i) in enumerate(zip(S, ratio), start=1):
            w.writerow([i, float(s_i), float(r_i)])
    print(f"saved {csv_path}")

    # Plot — no in-figure title (ICML rule)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(idx, ratio_p, "o-", color="#1f77b4", lw=1.3, ms=5)
    ax.set_yscale("log")
    ax.set_xlabel("singular-value index $i$")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.set_xticks(idx[::2])
    ax.set_ylim(floor, 2.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(1e-2, ls="--", lw=0.8, color="grey", alpha=0.7)
    ax.text(idx[-1], 1.1e-2, r"$10^{-2}$", color="grey",
            ha="right", va="bottom", fontsize=8)
    fig.tight_layout()

    pdf = args.out_dir / f"{args.stem}.pdf"
    png = args.out_dir / f"{args.stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {pdf}")
    print(f"saved {png}")

    cum = np.cumsum(S ** 2) / np.sum(S ** 2)
    print(f"sigma_1 = {S[0]:.4f}")
    print("sigma_i / sigma_1 (first 8): "
          + ", ".join(f"{r:.3g}" for r in ratio[:8]))
    print("cumulative variance fraction (k=1..8): "
          + ", ".join(f"{c:.3g}" for c in cum[:8]))


if __name__ == "__main__":
    main()
