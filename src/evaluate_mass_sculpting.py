#!/usr/bin/env python3
"""
Evaluate mass sculpting from inference results (ROOT or H5).

Produces:
  1. Mass distribution: jet soft-drop mass for various classifier cuts.
  2. JSD vs efficiency: Jensen–Shannon divergence of the mass distribution
     vs reference, as a function of efficiency.
  3. Classification accuracy: Hbb vs QCD accuracy on QCD and Hbb samples.

Usage:
  # ROOT (ParT/OGParT style: score_label_QCD, score_label_Hbb, jet_sdmass)
  python evaluate_mass_sculpting.py --inference-result /path/to/scores.root -o ./plots

  # H5 (scores array [:,0]=QCD, [:,1]=Hbb; jet_sdmass in same file or --mass-file)
  python evaluate_mass_sculpting.py --inference-result /path/to/scores.h5 -o ./plots
  python evaluate_mass_sculpting.py --inference-result /path/to/scores.h5 --mass-file /path/to/masses.root -o ./plots

  # Compare inference result vs OGParT on QCD (one mass-distribution plot per cut)
  python evaluate_mass_sculpting.py --inference-result /path/to/scores.h5 --ogpart /path/to/OGParT_on_QCD.root -o ./plots --tag PerpParT

  # Compare on both QCD and Hbb samples
  python evaluate_mass_sculpting.py --inference-result /path/to/scores_QCD.h5 --ogpart /path/to/OGParT_on_QCD.root --inference-result-hbb /path/to/scores_Hbb.h5 --ogpart-hbb /path/to/OGParT_on_Hbb.root -o ./plots --tag PerpParT
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Optional: uproot for ROOT, h5py for H5
try:
    import uproot
except ImportError:
    uproot = None
try:
    import h5py
except ImportError:
    h5py = None

# Classification threshold
CLASS_THRESHOLD = 0.5


def _to1d(a) -> np.ndarray:
    return np.asarray(a).flatten().astype(np.float64)


def _accuracy_qcd(score: np.ndarray, threshold: float = CLASS_THRESHOLD) -> float:
    """Accuracy on QCD: fraction correctly classified as QCD (score < threshold)."""
    return float((_to1d(score) < threshold).mean())


def _accuracy_hbb(score: np.ndarray, threshold: float = CLASS_THRESHOLD) -> float:
    """Accuracy on Hbb: fraction correctly classified as Hbb (score >= threshold)."""
    return float((_to1d(score) >= threshold).mean())


def _combined_accuracy(
    score_qcd: np.ndarray,
    score_hbb: np.ndarray,
    threshold: float = CLASS_THRESHOLD,
) -> tuple[float, float]:
    """
    Combined accuracy: returns (weighted, balanced).
    - Weighted: sample-size weighted average
    - Balanced: arithmetic mean of QCD and Hbb accuracies
    """
    acc_qcd = _accuracy_qcd(score_qcd, threshold)
    acc_hbb = _accuracy_hbb(score_hbb, threshold)
    n_qcd, n_hbb = len(score_qcd), len(score_hbb)
    weighted = float((n_qcd * acc_qcd + n_hbb * acc_hbb) / (n_qcd + n_hbb))
    balanced = float((acc_qcd + acc_hbb) / 2)
    return weighted, balanced


# ---------------------------------------------------------------------------
# JSD vs efficiency (from PerpParT_mass_sculpting.ipynb)
# ---------------------------------------------------------------------------

def _weighted_hist(m: np.ndarray, w: np.ndarray, bins: int, mrange: tuple[float, float]) -> np.ndarray:
    h, _ = np.histogram(m, bins=bins, range=mrange, weights=w)
    h = h.astype(np.float64)
    s = h.sum()
    return (h / s) if s > 0 else np.ones_like(h) / len(h)


def _jsd(p: np.ndarray, q: np.ndarray, eps: float = 1e-12, normalize: bool = True) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)))
    kl_qm = np.sum(q * (np.log(q) - np.log(m)))
    jsd = 0.5 * (kl_pm + kl_qm)
    return float(jsd / np.log(2.0) if normalize else jsd)


def jsd_vs_efficiency(
    mass: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray | None = None,
    *,
    is_background_mask: np.ndarray | None = None,
    bins: int = 60,
    mrange: tuple[float, float] | None = None,
    eff_grid: np.ndarray | None = None,
    reference: str = "global",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if eff_grid is None:
        eff_grid = np.linspace(0.9, 0.0005, 9)
    if is_background_mask is None:
        is_background_mask = np.ones(len(score), dtype=bool)
    m = _to1d(mass[is_background_mask])
    s = _to1d(score[is_background_mask])
    w = np.ones_like(m, dtype=np.float64) if weight is None else _to1d(weight[is_background_mask])

    if mrange is None:
        lo, hi = np.quantile(m, [0.001, 0.999])
        mrange = (float(lo), float(hi))

    if reference == "global":
        Q = _weighted_hist(m, w, bins, mrange)
    elif reference == "low":
        t_low = np.quantile(s, 0.5)
        sel = s <= t_low
        Q = _weighted_hist(m[sel], w[sel], bins, mrange)
    else:
        raise ValueError("reference must be 'global' or 'low'.")

    jsd_list, eff_list, thr_list = [], [], []
    for eff in eff_grid:
        t = np.quantile(s, 1.0 - eff)
        sel = s >= t
        P = _weighted_hist(m[sel], w[sel], bins, mrange)
        jsd_list.append(_jsd(P, Q))
        eff_list.append(eff)
        thr_list.append(t)
    return np.array(eff_list), np.array(thr_list), np.array(jsd_list)


# ---------------------------------------------------------------------------
# Load inference: ROOT or H5
# ---------------------------------------------------------------------------

def _load_root(
    path: str,
    *,
    tree: str = "Events;1",
    qcd_branch: str = "score_label_QCD",
    hbb_branch: str = "score_label_Hbb",
    mass_key: str = "jet_sdmass",
    load_mass: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    if uproot is None:
        raise RuntimeError("uproot is required for ROOT files. pip install uproot")
    f = uproot.open(path)
    t = f[tree]
    QCD = _to1d(t[qcd_branch].arrays().to_numpy())
    Xbb = _to1d(t[hbb_branch].arrays().to_numpy())
    score = Xbb / (Xbb + QCD + 1e-12)
    if not load_mass:
        return score, None
    jet_sdmass = _to1d(t[mass_key].arrays().to_numpy())
    return score, jet_sdmass


def _h5_find_key(f, keys: list[str]) -> str | None:
    for k in keys:
        if k in f:
            return k
    return None


def _load_h5(
    path: str,
    *,
    scores_path: str = "scores",
    mass_key: str = "jet_sdmass",
    mass_file: str | None = None,
    mass_tree: str = "Events;1",
    need_mass: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    if h5py is None:
        raise RuntimeError("h5py is required for HDF5 files. pip install h5py")
    with h5py.File(path, "r") as f:
        try:
            s = f[scores_path][:]
        except KeyError:
            raise KeyError(f"H5 {path}: scores not found at '{scores_path}'. Use --scores-path.")
        QCD = _to1d(s[:, 0])
        Xbb = _to1d(s[:, 1])
        score = Xbb / (Xbb + QCD + 1e-12)

        if not need_mass:
            return score, None

        # Mass: from same H5 or --mass-file
        mass = None
        for key in (mass_key, "jet_sdmass", "mass", "m"):
            if key in f:
                mass = _to1d(f[key][:])
                break
        if mass is not None:
            return score, mass

    if mass_file is None:
        raise ValueError(
            f"H5 {path} has no 'jet_sdmass'/'mass'/'m'. Provide --mass-file (ROOT or H5) with jet masses."
        )

    # Load mass from --mass-file
    p = Path(mass_file)
    if p.suffix.lower() in (".root",):
        if uproot is None:
            raise RuntimeError("uproot is required to read mass from ROOT --mass-file.")
        rf = uproot.open(mass_file)
        t = rf[mass_tree]
        for key in (mass_key, "jet_sdmass", "mass"):
            if key in t.keys():
                mass = _to1d(t[key].arrays().to_numpy())
                break
        else:
            raise KeyError(f"ROOT {mass_file} tree {mass_tree}: no jet_sdmass/mass branch.")
    else:
        with h5py.File(mass_file, "r") as mf:
            k = _h5_find_key(mf, [mass_key, "jet_sdmass", "mass", "m"])
            if k is None:
                raise KeyError(f"H5 {mass_file}: no jet_sdmass/mass/m dataset.")
            mass = _to1d(mf[k][:])

    if len(mass) != len(score):
        raise ValueError(
            f"Length mismatch: inference scores n={len(score)}, mass n={len(mass)}. "
            "Mass file must match inference events."
        )
    return score, mass


def load_inference(
    path: str,
    *,
    tree: str = "Events;1",
    scores_path: str = "scores",
    mass_key: str = "jet_sdmass",
    mass_file: str | None = None,
    mass_tree: str = "Events;1",
    return_mass: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load Xbb/(Xbb+QCD) and optionally jet_sdmass from ROOT or H5. Returns (score, jet_sdmass or None)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".root":
        return _load_root(path, tree=tree, mass_key=mass_key, load_mass=return_mass)
    if p.suffix.lower() in (".h5", ".hdf5"):
        return _load_h5(
            path,
            scores_path=scores_path,
            mass_key=mass_key,
            mass_file=mass_file if return_mass else None,
            mass_tree=mass_tree,
            need_mass=return_mass,
        )
    raise ValueError(f"Unsupported format: {path}. Use .root or .h5.")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_mass_distribution(
    score: np.ndarray,
    jet_sdmass: np.ndarray,
    out_dir: str,
    *,
    cuts: list[float] = (0.5, 0.7, 0.9, 0.99, 0.995, 0.997, 0.999),
    bins: np.ndarray | None = None,
    tag: str = "Classifier",
    score_baseline: np.ndarray | None = None,
    tag_baseline: str | None = None,
    acc_inference: float | None = None,
    acc_baseline: float | None = None,
    sample_type: str = "QCD",
) -> list[str]:
    if bins is None:
        bins = np.arange(0, 310, 10)
    compare = score_baseline is not None and tag_baseline is not None
    saved: list[str] = []

    if compare:
        # One plot per cut: Data + inference vs OGParT
        for c in cuts:
            fig, ax = plt.subplots()
            ax.hist(jet_sdmass, bins=bins, histtype="step", label="Data", density=True, color="gray", zorder=0)
            mask = score > c
            mask_b = score_baseline > c
            ax.hist(jet_sdmass[mask], bins=bins, histtype="step", label=tag, density=True)
            ax.hist(jet_sdmass[mask_b], bins=bins, histtype="step", label=tag_baseline, density=True)
            ax.set_xlabel("Jet soft-drop mass [GeV]")
            ax.set_ylabel("Density")
            ax.legend(title=f"Cut: {c}")
            # Add accuracy text box
            if acc_inference is not None and acc_baseline is not None:
                acc_text = f"{tag}: {acc_inference:.3f}\n{tag_baseline}: {acc_baseline:.3f}"
            elif acc_inference is not None:
                acc_text = f"{tag}: {acc_inference:.3f}"
            else:
                acc_text = ""
            if acc_text:
                ax.text(
                    0.02, 0.02, f"Acc ({sample_type}):\n{acc_text}",
                    transform=ax.transAxes, va="bottom", ha="left",
                    fontsize=10, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
                )
            fn = f"mass_distribution_cut_{c}.png"
            out = os.path.join(out_dir, fn)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(out)
    else:
        # One plot with all cuts
        fig, ax = plt.subplots()
        ax.hist(jet_sdmass, bins=bins, histtype="step", label="Data", density=True, color="gray", zorder=0)
        for c in cuts:
            mask = score > c
            ax.hist(jet_sdmass[mask], bins=bins, histtype="step", label=f"Cut: {c}", density=True)
        ax.set_xlabel("Jet soft-drop mass [GeV]")
        ax.set_ylabel("Density")
        ax.set_title(f"Mass distribution after classifier cut ({tag})")
        ax.legend()
        if acc_inference is not None:
            ax.text(
                0.02, 0.02, f"Acc ({sample_type}): {acc_inference:.3f}",
                transform=ax.transAxes, va="bottom", ha="left",
                fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
            )
        out = os.path.join(out_dir, "mass_distribution.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)
    return saved


def plot_jsd_vs_efficiency(
    score: np.ndarray,
    jet_sdmass: np.ndarray,
    out_dir: str,
    *,
    bins: int = 60,
    mrange: tuple[float, float] = (0.0, 300.0),
    eff_grid: np.ndarray | None = None,
    tag: str = "Classifier",
    score_baseline: np.ndarray | None = None,
    tag_baseline: str | None = None,
    acc_inference: float | None = None,
    acc_baseline: float | None = None,
    sample_type: str = "QCD",
) -> str:
    if eff_grid is None:
        eff_grid = np.linspace(0.9, 0.0005, 9)
    fig, ax = plt.subplots()
    eff, _thr, jsd_vals = jsd_vs_efficiency(
        mass=jet_sdmass,
        score=score,
        weight=None,
        bins=bins,
        mrange=mrange,
        eff_grid=eff_grid,
        reference="global",
    )
    ax.scatter(x=eff, y=jsd_vals, label=tag)
    if score_baseline is not None and tag_baseline is not None:
        _e, _t, jsd_b = jsd_vs_efficiency(
            mass=jet_sdmass,
            score=score_baseline,
            weight=None,
            bins=bins,
            mrange=mrange,
            eff_grid=eff_grid,
            reference="global",
        )
        ax.scatter(x=_e, y=jsd_b, label=tag_baseline)
    ax.set_xlabel("Efficiency")
    ax.set_ylabel("JSD (mass sculpting)")
    ax.set_title("Jensen–Shannon divergence vs efficiency")
    ax.legend()
    # Add accuracy text box
    if acc_inference is not None and acc_baseline is not None:
        acc_text = f"{tag}: {acc_inference:.3f}\n{tag_baseline}: {acc_baseline:.3f}"
    elif acc_inference is not None:
        acc_text = f"{tag}: {acc_inference:.3f}"
    else:
        acc_text = ""
    if acc_text:
        ax.text(
            0.02, 0.02, f"Acc ({sample_type}):\n{acc_text}",
            transform=ax.transAxes, va="bottom", ha="left",
            fontsize=10, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
        )
    out = os.path.join(out_dir, "jsd_vs_efficiency.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_roc_curve(
    score_qcd: np.ndarray,
    score_hbb: np.ndarray,
    score_qcd_baseline: np.ndarray | None,
    score_hbb_baseline: np.ndarray | None,
    out_dir: str,
    *,
    tag: str = "Classifier",
    tag_baseline: str = "OGParT",
    suffix: str = "",
) -> str:
    """Plot ROC curve for Hbb vs QCD classification (optimized O(n log n))."""
    # True labels: QCD=0, Hbb=1
    y_true = np.concatenate([np.zeros(len(score_qcd)), np.ones(len(score_hbb))])
    scores_inf = np.concatenate([score_qcd, score_hbb])
    
    n_pos = np.sum(y_true == 1)  # Hbb
    n_neg = np.sum(y_true == 0)  # QCD
    
    def _compute_roc(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute ROC curve efficiently using cumulative sums. Returns (fpr, tpr, thresholds, auc)."""
        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1]
        y_sorted = y[sorted_indices]
        scores_sorted = scores[sorted_indices]
        
        # Cumulative TP and FP: at threshold i, we predict top i as Hbb
        # TP = cumulative sum of Hbb labels (y==1) in sorted order
        # FP = cumulative sum of QCD labels (y==0) in sorted order
        tp_cumsum = np.cumsum(y_sorted)
        fp_cumsum = np.cumsum(1 - y_sorted)
        
        # Add point at (0,0) for threshold = all predicted as QCD
        tpr = np.concatenate([[0.0], tp_cumsum / n_pos if n_pos > 0 else tp_cumsum])
        fpr = np.concatenate([[0.0], fp_cumsum / n_neg if n_neg > 0 else fp_cumsum])
        # Thresholds: at index i, threshold is scores_sorted[i-1] (or inf for i=0)
        thresholds = np.concatenate([[np.inf], scores_sorted])
        
        # AUC using trapezoidal rule
        auc = np.trapz(tpr, fpr)
        return fpr, tpr, thresholds, auc
    
    # Compute ROC for inference
    fpr_inf, tpr_inf, thresh_inf, auc_inf = _compute_roc(scores_inf, y_true)
    
    fig, ax = plt.subplots()
    # Transposed: X=TPR, Y=FPR
    ax.plot(tpr_inf, fpr_inf, label=f"{tag} (AUC={auc_inf:.4f})", linewidth=2)
    
    # Baseline if available
    if score_qcd_baseline is not None and score_hbb_baseline is not None:
        scores_base = np.concatenate([score_qcd_baseline, score_hbb_baseline])
        fpr_base, tpr_base, thresh_base, auc_base = _compute_roc(scores_base, y_true)
        ax.plot(tpr_base, fpr_base, label=f"{tag_baseline} (AUC={auc_base:.4f})", linewidth=2)
    
    # Mark specific FPR thresholds
    fpr_markers = [0.001, 0.005, 0.01]  # 0.1%, 0.5%, 1%
    fpr_labels = ["0.1%", "0.5%", "1%"]
    
    def _find_score_at_fpr(fpr_target: float, fpr_arr: np.ndarray, tpr_arr: np.ndarray, thresh_arr: np.ndarray) -> tuple[float | None, float | None]:
        """Find TPR and score threshold at a specific FPR using interpolation."""
        # Find indices where FPR crosses the target
        idx = np.searchsorted(fpr_arr, fpr_target)
        if idx == 0:
            return (tpr_arr[0] if len(tpr_arr) > 0 else None, thresh_arr[0] if len(thresh_arr) > 0 else None)
        if idx >= len(fpr_arr):
            return (tpr_arr[-1] if len(tpr_arr) > 0 else None, thresh_arr[-1] if len(thresh_arr) > 0 else None)
        # If exact match
        if fpr_arr[idx] == fpr_target:
            return (tpr_arr[idx], thresh_arr[idx])
        # Interpolate TPR, but for threshold use the one at idx-1 (the point just below target FPR)
        # This is the threshold that gives us FPR <= target
        if idx > 0:
            t = (fpr_target - fpr_arr[idx - 1]) / (fpr_arr[idx] - fpr_arr[idx - 1])
            tpr_val = tpr_arr[idx - 1] + t * (tpr_arr[idx] - tpr_arr[idx - 1])
            # Use threshold from idx-1 (lower FPR = higher threshold = stricter cut)
            score_val = thresh_arr[idx - 1]
            return (tpr_val, score_val)
        return (tpr_arr[idx], thresh_arr[idx])
    
    # Mark points for inference with offset labels to avoid overlap
    offsets_y = [1.5, 2.0, 2.5]  # Different vertical offsets for each marker
    offsets_x = [-0.02, 0.0, 0.02]  # Slight horizontal offsets
    for i, (fpr_tgt, fpr_lbl) in enumerate(zip(fpr_markers, fpr_labels)):
        tpr_at_fpr, score_at_fpr = _find_score_at_fpr(fpr_tgt, fpr_inf, tpr_inf, thresh_inf)
        if tpr_at_fpr is not None and score_at_fpr is not None:
            ax.plot(tpr_at_fpr, fpr_tgt, "o", color="C0", markersize=8, zorder=5)
            ax.text(tpr_at_fpr + offsets_x[i], fpr_tgt * offsets_y[i], 
                   f"{fpr_lbl} FPR\nXbb={score_at_fpr:.4f}", 
                   ha="center", va="bottom", fontsize=7, color="C0",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="C0"))
    
    # Mark points for baseline if available (offset differently to avoid overlap)
    if score_qcd_baseline is not None and score_hbb_baseline is not None:
        offsets_y_base = [2.0, 2.5, 3.0]  # Different vertical offsets for baseline
        offsets_x_base = [0.02, 0.0, -0.02]  # Opposite horizontal offsets
        for i, (fpr_tgt, fpr_lbl) in enumerate(zip(fpr_markers, fpr_labels)):
            tpr_at_fpr, score_at_fpr = _find_score_at_fpr(fpr_tgt, fpr_base, tpr_base, thresh_base)
            if tpr_at_fpr is not None and score_at_fpr is not None:
                ax.plot(tpr_at_fpr, fpr_tgt, "s", color="C1", markersize=8, zorder=5)
                ax.text(tpr_at_fpr + offsets_x_base[i], fpr_tgt * offsets_y_base[i], 
                       f"{fpr_lbl} FPR\nscore={score_at_fpr:.4f}", 
                       ha="center", va="bottom", fontsize=7, color="C1",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="C1"))
    
    ax.set_xlabel("True Positive Rate (Hbb efficiency)")
    ax.set_ylabel("False Positive Rate (QCD → Hbb)")
    title = "ROC Curve: Hbb vs QCD (75-175 GeV)" if suffix == "_75_175" else "ROC Curve: Hbb vs QCD"
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_xlim([0.4, 1])
    ax.set_ylim([1e-4, 1])
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    out = os.path.join(out_dir, f"roc_curve{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# (subdir_name, mass_window (lo, hi) or None, mrange for JSD, bins for mass hist or None)
SUBDIRS = [
    ("full_jets", None, (0.0, 300.0), None),
    ("75_175_jets", (75, 175), (75.0, 175.0), np.arange(75, 176, 5)),
]


def _parse_cuts(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _process_sample(
    score: np.ndarray,
    jet_sdmass: np.ndarray,
    score_og: np.ndarray | None,
    sample_type: str,
    args,
    base_out: str,
) -> None:
    """Process one sample (QCD or Hbb): create plots for all subdirs."""
    for subdir_name, mass_window, mrange, bins_override in SUBDIRS:
        out = os.path.join(base_out, subdir_name)
        os.makedirs(out, exist_ok=True)
        if mass_window is not None:
            m = (jet_sdmass >= mass_window[0]) & (jet_sdmass <= mass_window[1])
            s, j = score[m], jet_sdmass[m]
            s_og = score_og[m] if score_og is not None else None
            print(f"  {sample_type} {subdir_name}: n={len(s)} jets in [{mass_window[0]}, {mass_window[1]}] GeV")
        else:
            s, j = score, jet_sdmass
            s_og = score_og
            print(f"  {sample_type} {subdir_name}: n={len(s)} jets")

        # Compute accuracies
        if sample_type == "QCD":
            acc_s = _accuracy_qcd(s) if s_og is None else _accuracy_qcd(s)
            acc_og = _accuracy_qcd(s_og) if s_og is not None else None
        else:  # Hbb
            acc_s = _accuracy_hbb(s) if s_og is None else _accuracy_hbb(s)
            acc_og = _accuracy_hbb(s_og) if s_og is not None else None

        mass_paths = plot_mass_distribution(
            s, j, out, cuts=args.cuts, tag=args.tag,
            score_baseline=s_og, tag_baseline=args.ogpart_tag if s_og is not None else None,
            bins=bins_override,
            acc_inference=acc_s, acc_baseline=acc_og,
            sample_type=sample_type,
        )
        for p in mass_paths:
            print(f"Saved {p}")
        p2 = plot_jsd_vs_efficiency(
            s, j, out, mrange=mrange, tag=args.tag,
            score_baseline=s_og, tag_baseline=args.ogpart_tag if s_og is not None else None,
            acc_inference=acc_s, acc_baseline=acc_og,
            sample_type=sample_type,
        )
        print(f"Saved {p2}")


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate mass sculpting from inference (ROOT or H5). Outputs mass distribution and JSD vs efficiency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--inference-result", required=True, dest="inference_result", help="Path to QCD inference result: .root (ParT/OGParT) or .h5 (scores[:,0]=QCD, [:,1]=Hbb)")
    ap.add_argument("--inference-result-hbb", default=None, dest="inference_result_hbb", help="Path to Hbb inference result H5 (optional)")
    ap.add_argument("-o", "--output-dir", required=True, dest="output_dir", help="Directory to save plots")
    # ROOT
    ap.add_argument("--tree", default="Events;1", help="ROOT tree name")
    ap.add_argument("--mass-key", default="jet_sdmass", help="Branch/dataset name for jet soft-drop mass")
    # H5
    ap.add_argument("--scores-path", default="scores", help="H5 path to scores dataset (e.g. 'scores' or 'infer/scores')")
    ap.add_argument("--mass-file", default=None, help="Path to ROOT or H5 with jet masses when not in inference-result H5 (ignored if --ogpart)")
    ap.add_argument("--mass-tree", default="Events;1", help="ROOT tree for --mass-file when it is .root")
    # Comparison with OGParT
    ap.add_argument("--ogpart", default=None, help="Path to OGParT/ParT ROOT for QCD comparison; uses its jet_sdmass for both (--mass-file ignored)")
    ap.add_argument("--ogpart-hbb", default=None, dest="ogpart_hbb", help="Path to OGParT/ParT ROOT for Hbb comparison")
    ap.add_argument("--ogpart-tag", default="OGParT", dest="ogpart_tag", help="Label for OGParT in plots")
    # Plot options
    ap.add_argument("--cuts", type=_parse_cuts, default="0.5,0.7,0.9,0.99,0.995,0.997,0.999", help="Comma-separated classifier cuts for mass distribution")
    ap.add_argument("--tag", default="Classifier", help="Label for inference result in legend")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_out = os.path.join(args.output_dir, args.tag)

    # Process QCD sample
    score_og_qcd: np.ndarray | None = None
    if args.ogpart is not None:
        score_og_qcd, jet_sdmass_qcd = _load_root(args.ogpart, tree=args.tree, mass_key=args.mass_key, load_mass=True)
        score_qcd, _ = load_inference(
            args.inference_result,
            tree=args.tree,
            scores_path=args.scores_path,
            mass_key=args.mass_key,
            mass_file=None,
            mass_tree=args.mass_tree,
            return_mass=False,
        )
        if len(score_qcd) != len(jet_sdmass_qcd):
            raise ValueError(
                f"Length mismatch: OGParT QCD n={len(jet_sdmass_qcd)}, inference QCD n={len(score_qcd)}. "
                "Both must be run on the same QCD sample in the same order."
            )
        print(f"QCD Comparison: n={len(jet_sdmass_qcd)}, OGParT score [{score_og_qcd.min():.4f}, {score_og_qcd.max():.4f}], inference [{score_qcd.min():.4f}, {score_qcd.max():.4f}], jet_sdmass [{jet_sdmass_qcd.min():.2f}, {jet_sdmass_qcd.max():.2f}] GeV")
    else:
        score_qcd, jet_sdmass_qcd = load_inference(
            args.inference_result,
            tree=args.tree,
            scores_path=args.scores_path,
            mass_key=args.mass_key,
            mass_file=args.mass_file,
            mass_tree=args.mass_tree,
        )
        assert jet_sdmass_qcd is not None
        print(f"QCD Loaded: n={len(score_qcd)} events, score [{score_qcd.min():.4f}, {score_qcd.max():.4f}], jet_sdmass [{jet_sdmass_qcd.min():.2f}, {jet_sdmass_qcd.max():.2f}] GeV")

    _process_sample(score_qcd, jet_sdmass_qcd, score_og_qcd, "QCD", args, os.path.join(base_out, "QCD"))

    # Process Hbb sample if provided
    if args.inference_result_hbb is not None:
        score_og_hbb: np.ndarray | None = None
        if args.ogpart_hbb is not None:
            score_og_hbb, jet_sdmass_hbb = _load_root(args.ogpart_hbb, tree=args.tree, mass_key=args.mass_key, load_mass=True)
            score_hbb, _ = load_inference(
                args.inference_result_hbb,
                tree=args.tree,
                scores_path=args.scores_path,
                mass_key=args.mass_key,
                mass_file=None,
                mass_tree=args.mass_tree,
                return_mass=False,
            )
            if len(score_hbb) != len(jet_sdmass_hbb):
                raise ValueError(
                    f"Length mismatch: OGParT Hbb n={len(jet_sdmass_hbb)}, inference Hbb n={len(score_hbb)}. "
                    "Both must be run on the same Hbb sample in the same order."
                )
            print(f"Hbb Comparison: n={len(jet_sdmass_hbb)}, OGParT score [{score_og_hbb.min():.4f}, {score_og_hbb.max():.4f}], inference [{score_hbb.min():.4f}, {score_hbb.max():.4f}], jet_sdmass [{jet_sdmass_hbb.min():.2f}, {jet_sdmass_hbb.max():.2f}] GeV")
        else:
            score_hbb, jet_sdmass_hbb = load_inference(
                args.inference_result_hbb,
                tree=args.tree,
                scores_path=args.scores_path,
                mass_key=args.mass_key,
                mass_file=None,
                mass_tree=args.mass_tree,
            )
            assert jet_sdmass_hbb is not None
            print(f"Hbb Loaded: n={len(score_hbb)} events, score [{score_hbb.min():.4f}, {score_hbb.max():.4f}], jet_sdmass [{jet_sdmass_hbb.min():.2f}, {jet_sdmass_hbb.max():.2f}] GeV")

        _process_sample(score_hbb, jet_sdmass_hbb, score_og_hbb, "Hbb", args, os.path.join(base_out, "Hbb"))

        # Print combined accuracy and plot ROC if both samples available
        if args.ogpart is not None and args.ogpart_hbb is not None:
            acc_qcd_inf = _accuracy_qcd(score_qcd)
            acc_hbb_inf = _accuracy_hbb(score_hbb)
            acc_qcd_og = _accuracy_qcd(score_og_qcd)
            acc_hbb_og = _accuracy_hbb(score_og_hbb)
            acc_w_inf, acc_b_inf = _combined_accuracy(score_qcd, score_hbb)
            acc_w_og, acc_b_og = _combined_accuracy(score_og_qcd, score_og_hbb)
            n_qcd, n_hbb = len(score_qcd), len(score_hbb)
            print(f"\nAccuracy Breakdown (QCD+Hbb):")
            print(f"  QCD (n={n_qcd}): {args.tag}={acc_qcd_inf:.4f}, {args.ogpart_tag}={acc_qcd_og:.4f}")
            print(f"  Hbb (n={n_hbb}): {args.tag}={acc_hbb_inf:.4f}, {args.ogpart_tag}={acc_hbb_og:.4f}")
            print(f"  Weighted (sample-size weighted, n_qcd={n_qcd}, n_hbb={n_hbb}):")
            print(f"    {args.tag}: {acc_w_inf:.4f}")
            print(f"    {args.ogpart_tag}: {acc_w_og:.4f}")
            print(f"  Balanced (arithmetic mean):")
            print(f"    {args.tag}: {acc_b_inf:.4f}")
            print(f"    {args.ogpart_tag}: {acc_b_og:.4f}")
            if n_qcd == n_hbb:
                print(f"  Note: Weighted = Balanced when n_qcd == n_hbb")
            
            # Plot ROC curve (full jets)
            roc_path = plot_roc_curve(
                score_qcd, score_hbb,
                score_og_qcd, score_og_hbb,
                base_out,
                tag=args.tag,
                tag_baseline=args.ogpart_tag,
            )
            print(f"Saved {roc_path}")
            
            # Plot ROC curve for 75-175 GeV jets
            mass_window = (75, 175)
            m_qcd = (jet_sdmass_qcd >= mass_window[0]) & (jet_sdmass_qcd <= mass_window[1])
            m_hbb = (jet_sdmass_hbb >= mass_window[0]) & (jet_sdmass_hbb <= mass_window[1])
            score_qcd_75_175 = score_qcd[m_qcd]
            score_hbb_75_175 = score_hbb[m_hbb]
            score_og_qcd_75_175 = score_og_qcd[m_qcd]
            score_og_hbb_75_175 = score_og_hbb[m_hbb]
            n_qcd_75_175, n_hbb_75_175 = len(score_qcd_75_175), len(score_hbb_75_175)
            print(f"\n75-175 GeV jets: QCD n={n_qcd_75_175}, Hbb n={n_hbb_75_175}")
            roc_path_75_175 = plot_roc_curve(
                score_qcd_75_175, score_hbb_75_175,
                score_og_qcd_75_175, score_og_hbb_75_175,
                base_out,
                tag=args.tag,
                tag_baseline=args.ogpart_tag,
                suffix="_75_175",
            )
            print(f"Saved {roc_path_75_175}")


if __name__ == "__main__":
    main()
