#!/usr/bin/env python3
"""Scree plot of singular values of M (the B x D matrix of centred QCD
class-conditional bin-mean CLS tokens).

Visual evidence (Claim 1) that the QCD jet-mass concept is approximately
low-rank in CLS space.

Reads the singular values stored in the S1 basis .npz produced by
``src/data_processing/build_mass_forget_bases.py``.

Outputs PNG + PDF in --out-dir.

Default basis: /scope-vol/svd_results/QCD_massshift_cls_tokens_ln_svd.npz
Default out:   /scope-vol/mass-sculpting-nulling/results/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--basis",
        type=Path,
        default=Path("/scope-vol/svd_results/QCD_massshift_cls_tokens_ln_svd.npz"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/scope-vol/mass-sculpting-nulling/results"),
    )
    ap.add_argument("--stem", default="scree_M")
    args = ap.parse_args()

    d = np.load(args.basis)
    s = np.asarray(d["s"], dtype=np.float64)
    n_bins = int(d["n_bins"])

    if s.size == 0 or s[0] == 0:
        raise SystemExit(f"basis has no usable singular values: {args.basis}")

    ratio = s / s[0]
    # rank(M) <= B-1 after centering; drop trailing numerically-zero tail.
    floor = 1e-7
    keep = ratio > floor
    ratio_p = ratio[keep]
    idx = np.arange(1, ratio_p.size + 1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(idx, ratio_p, "o-", color="#1f77b4", lw=1.3, ms=5)
    ax.set_yscale("log")
    ax.set_xlabel("singular-value index $i$")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.set_title(
        rf"Scree plot of $M$ (QCD class-conditional bin means, $B={n_bins}$)"
    )
    ax.set_xticks(idx[::2])
    ax.set_ylim(floor, 2.0)
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(1e-2, ls="--", lw=0.8, color="grey", alpha=0.7)
    ax.text(idx[-1], 1.1e-2, r"$10^{-2}$", color="grey", ha="right", va="bottom",
            fontsize=8)
    fig.tight_layout()

    png = args.out_dir / f"{args.stem}.png"
    pdf = args.out_dir / f"{args.stem}.pdf"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    cum = np.cumsum(s ** 2) / np.sum(s ** 2)
    print(f"saved {png}")
    print(f"saved {pdf}")
    print(f"sigma_1 = {s[0]:.4f}")
    print("sigma_i / sigma_1 (first 8): "
          + ", ".join(f"{r:.3g}" for r in ratio[:8]))
    print("cumulative variance fraction (k=1..8): "
          + ", ".join(f"{c:.3g}" for c in cum[:8]))


if __name__ == "__main__":
    main()
