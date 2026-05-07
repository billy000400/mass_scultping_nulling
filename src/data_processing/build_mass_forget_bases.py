#!/usr/bin/env python3
"""Build candidate "forget" subspaces for CURE+DisCo.

Produces three ``.npz`` files with the same schema as the existing SVD bases
(compatible with ``core/projections.py::load_svd_basis``):

    S1  QCD_massshift_cls_tokens_ln_svd.npz
        Class-mean PCA of QCD latents binned by jet mass.
        Top-k directions = "QCD mass-shift subspace".
    S2  QCD_massshift_minusHbb_cls_tokens_ln_svd.npz
        S1 after projecting out Hbb's class-mean PCA subspace.
        "QCD-unique mass-shift" -- directions along which QCD mean shifts with
        mass but Hbb doesn't. Theoretical sweet spot: removes QCD mass
        sensitivity while preserving most Hbb mass info.
    S3  QCD_olsmass_cls_tokens_ln_svd.npz
        Iteratively-deflated OLS mass-predictor directions on QCD.
        Rank-k "best linear mass predictor subspace" for QCD.

Each file has keys: ``Vh`` (kmax, D) orthonormal rows, ``mean`` (D,) = QCD
overall mean, ``s`` (kmax,) pseudo-singular-values used by variance-based k
selection, plus diagnostic scalars.

Also produces a sanity report:
    lda_results/forget_bases_r2.csv
    lda_results/forget_bases_r2.png

showing R^2(mass | latent) after nulling the top-k of each basis, for QCD
and Hbb separately, as k varies.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

QCD_LABEL = 0
HBB_LABEL = 1


# ---------- loaders / helpers ----------

def _load(h5_path: Path, latent_key: str, max_samples: int | None):
    with h5py.File(h5_path, "r") as hf:
        n_total = hf[latent_key].shape[0]
        n = min(n_total, max_samples) if max_samples is not None else n_total
        X = np.asarray(hf[latent_key][:n], dtype=np.float64).reshape(n, -1)
        y = np.asarray(hf["label"][:n]).reshape(-1).astype(np.int64)
        m = np.asarray(hf["jet_sdmass"][:n]).reshape(-1).astype(np.float64)
    return X, y, m


def _quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return np.clip(np.digitize(values, edges) - 1, 0, n_bins - 1)


def _orthonormalise(rows: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Row-wise Gram-Schmidt (drops rows that lose all norm)."""
    D = rows.shape[1]
    out = []
    for r in rows:
        v = r.copy()
        for u in out:
            v = v - np.dot(v, u) * u
        n = np.linalg.norm(v)
        if n > tol:
            out.append(v / n)
    if not out:
        return np.zeros((0, D))
    return np.stack(out, axis=0)


# ---------- strategies ----------

def build_s1_qcd_massshift(Xq: np.ndarray, mq: np.ndarray, n_bins: int,
                           kmax: int) -> dict:
    """Class-mean PCA of QCD binned by mass: Vh rows = top-k mass-shift directions."""
    bins = _quantile_bin(mq, n_bins)
    means = np.stack([Xq[bins == b].mean(0) for b in range(n_bins)], axis=0)
    qcd_mean = Xq.mean(0)
    centred = means - qcd_mean  # centre by QCD overall mean
    # SVD of centred means (n_bins, D)
    _, S, Vt = np.linalg.svd(centred, full_matrices=False)
    Vh = Vt[: min(kmax, len(S))]
    s = S[: Vh.shape[0]]
    return dict(Vh=Vh, s=s, mean=qcd_mean, n_bins=n_bins, strategy="S1_qcd_massshift")


def build_s2_qcd_unique_massshift(Xq: np.ndarray, mq: np.ndarray,
                                  Xh: np.ndarray, mh: np.ndarray,
                                  n_bins: int, kmax: int) -> dict:
    """S1 with Hbb's class-mean PCA subspace projected out first.

    Steps:
      1. Compute QCD class-mean PCA directions (rank <= n_bins-1).
      2. Compute Hbb class-mean PCA directions.
      3. Gram-Schmidt-remove Hbb's span from each QCD direction (reweighted by QCD
         singular values), then SVD the residuals to reorder by residual mass power.
    """
    b_q = _quantile_bin(mq, n_bins)
    means_q = np.stack([Xq[b_q == b].mean(0) for b in range(n_bins)], 0)
    qcd_mean = Xq.mean(0)
    c_q = means_q - qcd_mean
    _, sq, Vq = np.linalg.svd(c_q, full_matrices=False)  # Vq: (<=nbins, D)

    b_h = _quantile_bin(mh, n_bins)
    means_h = np.stack([Xh[b_h == b].mean(0) for b in range(n_bins)], 0)
    hbb_mean = Xh.mean(0)
    c_h = means_h - hbb_mean
    _, sh, Vh = np.linalg.svd(c_h, full_matrices=False)  # Vh: (<=nbins, D)

    # Orthonormal basis for span(Vh) (rows are already orthonormal from SVD).
    Qh = Vh  # (k_h, D)
    # Remove Hbb span from each QCD direction (scaled by its singular value so
    # early SVD rows retain more weight in the residual re-SVD).
    P = np.eye(Qh.shape[1]) - Qh.T @ Qh   # (D, D)
    resid = (Vq * sq[:, None]) @ P        # (<=nbins, D)
    # Re-SVD the residual rows to get ordered orthonormal basis.
    _, s_res, V_res = np.linalg.svd(resid, full_matrices=False)
    # Drop numerically-zero rows (no unique mass direction for that component).
    keep = s_res > (s_res.max() * 1e-6 if s_res.max() > 0 else 1e-12)
    V_res = V_res[keep]
    s_res = s_res[keep]

    kkeep = min(kmax, V_res.shape[0])
    return dict(Vh=V_res[:kkeep], s=s_res[:kkeep], mean=qcd_mean,
                n_bins=n_bins, strategy="S2_qcd_unique_massshift",
                n_unique=V_res.shape[0])


def build_s3_qcd_ols_deflated(Xq: np.ndarray, mq: np.ndarray, kmax: int,
                              ridge: float = 1e-3) -> dict:
    """Iteratively deflated OLS mass-predictor directions on QCD.

    At step i, fit w_i = argmin || X_res w - mass_c ||^2; record the R^2 explained
    by that unit direction; project it out of X_res; repeat.

    Equivalence to RAV (Regression Activation Vector): row 0 of the returned
    basis is exactly the unit-normalised ridge OLS coefficient w/||w|| of
    z-scored ``jet_sdmass`` regressed on QCD post-LN CLS tokens (centred by
    ``qcd_mean``). That is the standard "RAV" construction. Higher rows
    extend RAV to rank > 1 by iteratively deflating the chosen direction
    out of the residual data and re-fitting — the natural rank-k
    generalisation when the user wants a multi-dimensional removal subspace
    rather than a single direction.

    Caveat for downstream nulling: w/||w|| is the *coefficient* direction,
    not the direction along which mass varies most in input space. Because
    (X^T X + ridge I)^-1 inflates low-variance eigendirections, the unit
    coefficient often lives in a small-variance subspace. Projecting it
    out of X therefore barely changes X, and a linear probe on the
    residual still recovers most of the mass info. See the linear-probe
    sweep in ``scripts/probe_r2_vs_k.py`` for an empirical demonstration.
    """
    Xq = Xq.astype(np.float64)
    qcd_mean = Xq.mean(0)
    Xc = Xq - qcd_mean
    mc = (mq - mq.mean()) / (mq.std() + 1e-12)
    D = Xc.shape[1]

    directions = []
    r2_vals = []
    X_res = Xc.copy()
    for _ in range(kmax):
        A = X_res.T @ X_res + ridge * np.eye(D)
        b = X_res.T @ mc
        w = np.linalg.solve(A, b)
        nrm = np.linalg.norm(w)
        if nrm < 1e-9:
            break
        w = w / nrm
        # R^2 of this direction alone
        s = X_res @ w
        s_norm = (s - s.mean()) / (s.std() + 1e-12)
        r2 = float(np.corrcoef(s_norm, mc)[0, 1]) ** 2
        directions.append(w)
        r2_vals.append(r2)
        # Deflate: project w out of X_res
        X_res = X_res - np.outer(X_res @ w, w)

    V = np.stack(directions, axis=0) if directions else np.zeros((0, D))
    V = _orthonormalise(V)
    s = np.asarray(r2_vals[: V.shape[0]], dtype=np.float64)
    return dict(Vh=V, s=s, mean=qcd_mean, n_bins=0, strategy="S3_qcd_ols_deflated")


# ---------- sanity: R^2(mass | latent) after nulling ----------

def _r2_linear(X: np.ndarray, mass: np.ndarray, ridge: float = 1e-3) -> float:
    """Best linear R^2 for predicting mass from X (closed-form OLS with tiny ridge)."""
    Xc = X - X.mean(0, keepdims=True)
    mc = (mass - mass.mean()) / (mass.std() + 1e-12)
    A = Xc.T @ Xc + ridge * np.eye(Xc.shape[1])
    w = np.linalg.solve(A, Xc.T @ mc)
    pred = Xc @ w
    return float(np.corrcoef(pred, mc)[0, 1]) ** 2


def r2_after_null(X: np.ndarray, mass: np.ndarray,
                  Vh: np.ndarray, mean: np.ndarray, k: int) -> float:
    """Null the top-k rows of Vh (centred by mean) and report residual linear R^2."""
    if k <= 0 or Vh.shape[0] == 0:
        return _r2_linear(X, mass)
    V = Vh[:k]
    Xc = X - mean
    proj = (Xc @ V.T) @ V
    X_res = X - proj
    return _r2_linear(X_res, mass)


# ---------- sanity plot + csv ----------

def sanity_report(
    Xq: np.ndarray, mq: np.ndarray, Xh: np.ndarray, mh: np.ndarray,
    bases: dict[str, dict],
    k_grid: list[int],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    r2_q_full = _r2_linear(Xq, mq)
    r2_h_full = _r2_linear(Xh, mh)
    rows = [{"strategy": "baseline", "k": 0,
             "r2_qcd": r2_q_full, "r2_hbb": r2_h_full}]
    logger.info("baseline R^2(mass) QCD=%.3f Hbb=%.3f", r2_q_full, r2_h_full)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

    for name, basis in bases.items():
        Vh = np.asarray(basis["Vh"], dtype=np.float64)
        mean = np.asarray(basis["mean"], dtype=np.float64)
        kmax_b = Vh.shape[0]
        ks = [k for k in k_grid if k <= kmax_b]
        r2q = [r2_after_null(Xq, mq, Vh, mean, k) for k in ks]
        r2h = [r2_after_null(Xh, mh, Vh, mean, k) for k in ks]
        for k, q, h in zip(ks, r2q, r2h):
            rows.append({"strategy": name, "k": k, "r2_qcd": q, "r2_hbb": h})
            logger.info("%s k=%d   R^2 QCD=%.3f   R^2 Hbb=%.3f", name, k, q, h)
        axes[0].plot(ks, r2q, "o-", label=name)
        axes[1].plot(ks, r2h, "o-", label=name)

    for ax, label in zip(axes,
                         [f"QCD   (baseline {r2_q_full:.2f})",
                          f"Hbb   (baseline {r2_h_full:.2f})"]):
        ax.set_xlabel("k (nulled dim)")
        ax.set_ylabel(r"$R^2$(mass | latent)")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Residual mass-predictability after nulling top-k of each forget basis")
    fig.tight_layout()
    fig.savefig(out_dir / "forget_bases_r2.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "forget_bases_r2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["strategy", "k", "r2_qcd", "r2_hbb"])
        w.writeheader()
        w.writerows(rows)
    logger.info("Saved sanity report to %s", out_dir)


# ---------- save in CURE-compatible .npz schema ----------

def save_basis(basis: dict, out_path: Path, kmax_cap: int = 30) -> None:
    Vh = np.asarray(basis["Vh"], dtype=np.float64)
    s = np.asarray(basis["s"], dtype=np.float64)
    mean = np.asarray(basis["mean"], dtype=np.float64)

    keep = min(kmax_cap, Vh.shape[0])
    Vh = Vh[:keep]
    s = s[:keep]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        Vh=Vh, s=s, mean=mean,
        n_samples=np.int64(0),
        n_features=np.int64(mean.size),
        class_filter=np.int64(QCD_LABEL),
        class_filter_name=np.array("QCD", dtype="U"),
        strategy=np.array(basis.get("strategy", ""), dtype="U"),
        n_bins=np.int64(basis.get("n_bins", 0)),
    )
    logger.info("Saved %s  (Vh %s, s %s)", out_path, Vh.shape, s.shape)


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--h5", type=Path,
        default=Path("/scope-vol/hidden_out/train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5"),
    )
    ap.add_argument("--latent-key", default="cls_tokens_ln")
    ap.add_argument("--n-bins", type=int, default=20,
                    help="quantile mass bins per class (upper bound on subspace rank)")
    ap.add_argument("--kmax", type=int, default=20,
                    help="max number of basis rows to save per file")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("/scope-vol/svd_results"))
    ap.add_argument("--sanity-dir", type=Path,
                    default=Path("/scope-vol/lda_results/forget_bases"))
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    logger.info("Loading %s [%s]", args.h5, args.latent_key)
    X, y, m = _load(args.h5, args.latent_key, args.max_samples)
    logger.info("X=%s counts=%s", X.shape,
                dict(zip(*np.unique(y, return_counts=True))))
    Xq, mq = X[y == QCD_LABEL], m[y == QCD_LABEL]
    Xh, mh = X[y == HBB_LABEL], m[y == HBB_LABEL]

    logger.info("Building S1 (QCD mass-shift PCA)")
    s1 = build_s1_qcd_massshift(Xq, mq, args.n_bins, args.kmax)
    logger.info("Building S2 (QCD-unique mass-shift)")
    s2 = build_s2_qcd_unique_massshift(Xq, mq, Xh, mh, args.n_bins, args.kmax)
    logger.info("  S2 effective unique rank: %d", s2.get("n_unique", -1))
    logger.info("Building S3 (QCD OLS deflated)")
    s3 = build_s3_qcd_ols_deflated(Xq, mq, args.kmax)

    save_basis(s1, args.output_dir / "QCD_massshift_cls_tokens_ln_svd.npz",
               kmax_cap=args.kmax)
    save_basis(s2, args.output_dir / "QCD_massshift_minusHbb_cls_tokens_ln_svd.npz",
               kmax_cap=args.kmax)
    save_basis(s3, args.output_dir / "QCD_olsmass_cls_tokens_ln_svd.npz",
               kmax_cap=args.kmax)

    # Sanity: R^2 after nulling, for S1/S2/S3 and S4 (existing QCD SVD).
    s4_path = Path("/scope-vol/svd_results/QCD_res_cls_tokens_ln_svd.npz")
    bases = {"S1_massshift": s1, "S2_massshift_minusHbb": s2, "S3_ols_deflated": s3}
    if s4_path.exists():
        d4 = np.load(s4_path)
        bases["S4_full_qcd_svd"] = {"Vh": d4["Vh"], "s": d4["s"], "mean": d4["mean"]}

    k_grid = [1, 2, 3, 5, 7, 10, 15, 20]
    sanity_report(Xq, mq, Xh, mh, bases, k_grid, args.sanity_dir)


if __name__ == "__main__":
    main()
