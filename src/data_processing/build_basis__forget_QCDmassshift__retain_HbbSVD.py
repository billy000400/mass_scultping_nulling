#!/usr/bin/env python3
"""Build a forget basis that removes QCD mass-shift directions while retaining
the leading Hbb SVD signal subspace.

Strategy:
  1. Load the QCD mass-shift forget basis (S1) -- directions along which the QCD
     class-conditional mean shifts with jet mass.
  2. Load the Hbb SVD basis -- leading principal directions of Hbb sample variance.
  3. Gram-Schmidt each S1 direction against the top-k_hbb Hbb SVD directions,
     so that the resulting forget projection cannot remove components that lie
     in the Hbb signal subspace.
  4. Re-SVD the residual rows to get an ordered orthonormal basis; drop near-zero rows.
  5. Save as a CURE-compatible .npz:
       Vh    (k_out, D)  -- orthonormal forget directions
       s     (k_out,)    -- residual singular values (proxy for mass-shift power)
       mean  (D,)        -- QCD mean (from S1, used for centering in CURE projection)

Output: /scope-vol/svd_results/forget_QCDmassshift_retain_HbbSVD_cls_tokens_ln_svd.npz

Sweep --k-hbb to control how aggressively Hbb signal is protected.
Sweep --k-forget (passed to --svd-k at train time) to control how many
forget directions are applied.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_S1   = Path("/scope-vol/svd_results/QCD_massshift_cls_tokens_ln_svd.npz")
DEFAULT_HBB  = Path("/scope-vol/svd_results/Hbb_res_cls_tokens_ln_svd.npz")
DEFAULT_OUT  = Path("/scope-vol/svd_results/forget_QCDmassshift_retain_HbbSVD_cls_tokens_ln_svd.npz")


def gram_schmidt_remove(rows: np.ndarray, protect: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Project each row in `rows` orthogonal to the span of `protect` rows.

    `protect` rows are assumed orthonormal (e.g. from SVD).
    Returns residual rows (same shape, some may be near-zero).
    """
    # Project out the protected subspace: residual = row - (row @ P^T) P
    # where P = protect (k_hbb, D), orthonormal rows
    proj = rows @ protect.T  # (n_rows, k_hbb)
    return rows - proj @ protect  # (n_rows, D)


def build_basis(
    s1_path: Path,
    hbb_path: Path,
    k_hbb: int,
    kmax_out: int,
    tol: float = 1e-6,
) -> dict:
    s1  = np.load(s1_path)
    hbb = np.load(hbb_path)

    Vh_s1  = np.asarray(s1["Vh"],  dtype=np.float64)   # (k_s1, D)
    qcd_mean = np.asarray(s1["mean"], dtype=np.float64) # (D,)
    Vh_hbb = np.asarray(hbb["Vh"], dtype=np.float64)    # (128, D)

    k_hbb_use = min(k_hbb, Vh_hbb.shape[0])
    protect = Vh_hbb[:k_hbb_use]  # (k_hbb, D) -- orthonormal rows from SVD

    logger.info("S1 basis: %s  QCD mean: %s", Vh_s1.shape, qcd_mean.shape)
    logger.info("Hbb SVD basis: %s  using top k_hbb=%d", Vh_hbb.shape, k_hbb_use)

    # Log how much of each S1 direction is protected
    overlap = (Vh_s1 @ protect.T) ** 2  # (k_s1, k_hbb)
    for i, row_ov in enumerate(overlap[:5]):
        logger.info("  S1 dir %d: sum cos^2 with Hbb top-%d = %.4f  (max %.4f)",
                    i, k_hbb_use, row_ov.sum(), row_ov.max())

    # Remove Hbb signal subspace from S1 directions
    residuals = gram_schmidt_remove(Vh_s1, protect, tol=tol)  # (k_s1, D)

    # Re-SVD residuals to get ordered orthonormal basis
    # Weight by original S1 singular values so high-mass-power directions stay first
    s1_s = np.asarray(s1["s"], dtype=np.float64)
    weighted = residuals * s1_s[:len(residuals), None]
    _, s_res, Vh_res = np.linalg.svd(weighted, full_matrices=False)

    # Drop near-zero rows (directions that were entirely within Hbb signal subspace)
    threshold = s_res.max() * tol if s_res.max() > 0 else tol
    keep = s_res > threshold
    Vh_res = Vh_res[keep]
    s_res  = s_res[keep]
    logger.info("Residual basis after removing Hbb top-%d: %d directions kept "
                "(dropped %d)", k_hbb_use, keep.sum(), (~keep).sum())

    k_out = min(kmax_out, Vh_res.shape[0])
    return dict(
        Vh=Vh_res[:k_out],
        s=s_res[:k_out],
        mean=qcd_mean,
        k_hbb=k_hbb_use,
        strategy="forget_QCDmassshift_retain_HbbSVD",
    )


def save_basis(basis: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        Vh=basis["Vh"].astype(np.float64),
        s=basis["s"].astype(np.float64),
        mean=basis["mean"].astype(np.float64),
        k_hbb=np.int64(basis["k_hbb"]),
        strategy=np.array(basis["strategy"], dtype="U"),
    )
    logger.info("Saved: %s  Vh=%s  s=%s", out_path, basis["Vh"].shape, basis["s"].shape)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s1",      type=Path, default=DEFAULT_S1,
                    help="QCD mass-shift forget basis (S1) .npz")
    ap.add_argument("--hbb-svd", type=Path, default=DEFAULT_HBB,
                    help="Hbb SVD basis .npz (from svd_latent.py --class-filter Hbb)")
    ap.add_argument("--k-hbb",   type=int,  default=10,
                    help="Number of leading Hbb SVD directions to protect (default 10)")
    ap.add_argument("--kmax-out",type=int,  default=20,
                    help="Max rows to keep in output basis (default 20)")
    ap.add_argument("--out",     type=Path, default=DEFAULT_OUT,
                    help="Output .npz path")
    args = ap.parse_args()

    basis = build_basis(args.s1, args.hbb_svd, args.k_hbb, args.kmax_out)
    save_basis(basis, args.out)

    logger.info("Done. Use with: --svd-forget %s --svd-k <k>", args.out)
    logger.info("Recommended k range: 1..%d", basis["Vh"].shape[0])


if __name__ == "__main__":
    main()
