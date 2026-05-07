#!/usr/bin/env python3
"""Linear-probe R^2(jet_sdmass | h') vs k for four nulling-direction sources.

For each method and each k in --k-grid, project the top-k directions out of
the QCD post-LN CLS token and fit a ridge linear probe to predict
``jet_sdmass``. Report test R^2 on a held-out QCD split.

Methods:
    S1 (mass-binned PCA)  -- proposed; basis from
        /scope-vol/svd_results/QCD_massshift_cls_tokens_ln_svd.npz
    RAV (iterative OLS deflation, == S3) -- the standard "Regression
        Activation Vector" reading of RAV: at each step, solve the ridge
        regression w = argmin || X w - mass ||^2 + lambda ||w||^2, normalise
        w to unit length, project that direction out of X, repeat. Basis from
        /scope-vol/svd_results/QCD_olsmass_cls_tokens_ln_svd.npz . NB: the
        precomputed RAV vectors at /scope-vol/ravs/sdmass_*.h5 have
        metadata R^2=0 / beta=0 (failed an internal quality threshold and
        were zeroed) and are intentionally not used; we recompute the
        coefficient direction from scratch as S3.
    plain PCA (non-class-conditional QCD SVD) -- captures variance, not
        mass-shift; basis from
        /scope-vol/svd_results/QCD_res_cls_tokens_ln_svd.npz
    random k-d subspace -- sanity baseline (orthonormal Gaussian, averaged
        over --n-random-seeds seeds)

Linear probe only: the downstream classifier head is
LayerNorm(frozen)+Linear, so only linearly-accessible mass info can sculpt
the score. Nonlinear residual mass is unreachable by a linear head.

Outputs (under --out-dir):
    r2_vs_k.csv  -- columns: method, k, r2_test
    r2_vs_k.png, .pdf
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

QCD_LABEL = 0
DEFAULT_K_GRID = [0, 1, 2, 4, 8, 16]


def _load_qcd(h5_path: Path, latent_key: str, max_qcd: int | None):
    with h5py.File(h5_path, "r") as hf:
        labels = np.asarray(hf["label"][:]).reshape(-1).astype(np.int64)
        idx = np.where(labels == QCD_LABEL)[0]
        if max_qcd is not None and idx.size > max_qcd:
            idx = idx[:max_qcd]
        # h5py fancy indexing requires sorted indices already (it does)
        X = np.asarray(hf[latent_key][idx]).reshape(idx.size, -1).astype(np.float64)
        m = np.asarray(hf["jet_sdmass"][idx]).reshape(-1).astype(np.float64)
    return X, m


def _ridge_r2_train_test(
    X_tr: np.ndarray, m_tr: np.ndarray,
    X_te: np.ndarray, m_te: np.ndarray,
    ridge: float = 1e-3,
) -> float:
    mu = X_tr.mean(0, keepdims=True)
    Xc_tr = X_tr - mu
    Xc_te = X_te - mu
    m_mean = m_tr.mean()
    m_std = m_tr.std() + 1e-12
    mc_tr = (m_tr - m_mean) / m_std
    A = Xc_tr.T @ Xc_tr + ridge * np.eye(Xc_tr.shape[1])
    b = Xc_tr.T @ mc_tr
    w = np.linalg.solve(A, b)
    pred_te = Xc_te @ w
    target_te = (m_te - m_mean) / m_std
    ss_res = float(np.sum((target_te - pred_te) ** 2))
    ss_tot = float(np.sum((target_te - target_te.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def _null_topk(X: np.ndarray, Vh: np.ndarray, mean: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or Vh.shape[0] == 0:
        return X
    V = Vh[:k]
    Xc = X - mean
    return X - (Xc @ V.T) @ V


def _random_orth(D: int, k: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal(size=(D, k))
    Q, _ = np.linalg.qr(A)
    return Q.T  # (k, D), orthonormal rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--h5", type=Path,
        default=Path("/scope-vol/hidden_out/train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5"),
    )
    ap.add_argument("--latent-key", default="cls_tokens_ln")
    ap.add_argument("--max-qcd", type=int, default=400_000,
                    help="cap on QCD samples loaded (split 50/50 train/test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-random-seeds", type=int, default=5)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument(
        "--basis-s1", type=Path,
        default=Path("/scope-vol/svd_results/QCD_massshift_cls_tokens_ln_svd.npz"),
    )
    ap.add_argument(
        "--basis-rav", type=Path,
        default=Path("/scope-vol/svd_results/QCD_olsmass_cls_tokens_ln_svd.npz"),
    )
    ap.add_argument(
        "--basis-plain", type=Path,
        default=Path("/scope-vol/svd_results/QCD_res_cls_tokens_ln_svd.npz"),
    )
    ap.add_argument("--k-grid", type=int, nargs="+", default=DEFAULT_K_GRID)
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path("/scope-vol/mass-sculpting-nulling/results"),
    )
    ap.add_argument("--stem", default="r2_vs_k")
    args = ap.parse_args()

    print(f"Loading QCD samples from {args.h5} [{args.latent_key}] ...")
    X, m = _load_qcd(args.h5, args.latent_key, args.max_qcd)
    print(f"  loaded {X.shape[0]} QCD jets, D={X.shape[1]}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(X.shape[0])
    half = X.shape[0] // 2
    tr_idx = perm[:half]
    te_idx = perm[half:]
    X_tr, m_tr = X[tr_idx], m[tr_idx]
    X_te, m_te = X[te_idx], m[te_idx]
    print(f"  split: train={X_tr.shape[0]}  test={X_te.shape[0]}")

    # baseline (k=0)
    r2_base = _ridge_r2_train_test(X_tr, m_tr, X_te, m_te, args.ridge)
    print(f"baseline R^2(mass | latent), no nulling: {r2_base:.4f}")

    bases = {}
    for label, path in [
        ("S1 (mass-binned PCA)", args.basis_s1),
        ("RAV (iterative)", args.basis_rav),
        ("plain PCA", args.basis_plain),
    ]:
        d = np.load(path)
        Vh = np.asarray(d["Vh"], dtype=np.float64)
        mean = np.asarray(d["mean"], dtype=np.float64)
        bases[label] = (Vh, mean)
        print(f"  {label}: Vh {Vh.shape} from {path.name}")

    rows = []
    rows.append({"method": "baseline", "k": 0, "r2_test": r2_base})

    D = X.shape[1]
    for k in args.k_grid:
        if k == 0:
            continue
        for label, (Vh, mean) in bases.items():
            kk = min(k, Vh.shape[0])
            X_tr_p = _null_topk(X_tr, Vh, mean, kk)
            X_te_p = _null_topk(X_te, Vh, mean, kk)
            r2 = _ridge_r2_train_test(X_tr_p, m_tr, X_te_p, m_te, args.ridge)
            rows.append({"method": label, "k": k, "r2_test": r2})
            print(f"  k={k:2d}  {label:24s} R^2={r2:.4f}")

        # random subspace, averaged over seeds
        seed_r2 = []
        rand_mean = X_tr.mean(0)
        for s in range(args.n_random_seeds):
            r_rng = np.random.default_rng(args.seed + 1000 + s)
            V_rand = _random_orth(D, k, r_rng)
            X_tr_p = _null_topk(X_tr, V_rand, rand_mean, k)
            X_te_p = _null_topk(X_te, V_rand, rand_mean, k)
            seed_r2.append(_ridge_r2_train_test(X_tr_p, m_tr, X_te_p, m_te, args.ridge))
        r2_rand = float(np.mean(seed_r2))
        r2_rand_std = float(np.std(seed_r2))
        rows.append({"method": "random", "k": k, "r2_test": r2_rand,
                     "r2_test_std": r2_rand_std})
        print(f"  k={k:2d}  {'random (mean of '+str(args.n_random_seeds)+')':24s} "
              f"R^2={r2_rand:.4f} ± {r2_rand_std:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"{args.stem}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "k", "r2_test", "r2_test_std"])
        w.writeheader()
        for r in rows:
            r.setdefault("r2_test_std", "")
            w.writerow(r)
    print(f"saved {csv_path}")

    # Plot
    method_order = ["S1 (mass-binned PCA)", "RAV (iterative)", "plain PCA", "random"]
    method_color = {
        "S1 (mass-binned PCA)": "#1f77b4",
        "RAV (iterative)":      "#ff7f0e",
        "plain PCA":            "#2ca02c",
        "random":               "#7f7f7f",
    }
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for name in method_order:
        # Anchor each method at (k=0, r2_base): k=0 means "no nulling",
        # which is identical across methods by construction.
        ks, r2s, errs = [0], [r2_base], [0.0]
        for r in rows:
            if r["method"] != name or r["k"] == 0:
                continue
            ks.append(r["k"])
            r2s.append(r["r2_test"])
            errs.append(r.get("r2_test_std") or 0.0)
        order = np.argsort(ks)
        ks = np.array(ks)[order]
        r2s = np.array(r2s)[order]
        errs = np.array(errs)[order]
        if name == "random" and np.any(errs > 0):
            ax.errorbar(ks, r2s, yerr=errs, fmt="s--",
                        color=method_color[name], lw=1.2, ms=5,
                        capsize=2, label=name)
        else:
            ax.plot(ks, r2s, "o-", color=method_color[name],
                    lw=1.4, ms=5, label=name)
    ax.axhline(r2_base, ls=":", lw=1.0, color="black", alpha=0.7,
               label=f"no nulling ({r2_base:.3f})")
    ax.set_xlabel("k (number of nulled directions)")
    ax.set_ylabel(r"$R^2$ of jet $m_{\mathrm{SD}}$ from $h'$ (linear probe, QCD test)")
    # No in-figure title; the LaTeX caption is the title (ICML rule).
    # Linear x-axis: the symlog version of this plot inserts a wide
    # gap between k=0 and k=1 (the linear region of symlog gets the
    # same visual width as a full log decade). Linear avoids that
    # while keeping the k=0 anchor at the left edge.
    ax.set_xticks([0, 1, 2, 4, 8, 16])
    ax.set_xlim(-0.4, 16.4)
    ax.set_ylim(-0.02, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    png = args.out_dir / f"{args.stem}.png"
    pdf = args.out_dir / f"{args.stem}.pdf"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png}")
    print(f"saved {pdf}")


if __name__ == "__main__":
    main()
