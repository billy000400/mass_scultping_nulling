"""Shared metrics: accuracy, mean JSD, and efficiency@1%."""

from typing import Optional

import numpy as np


def compute_efficiency_at_misid_rate(
    probs: np.ndarray,
    targets: np.ndarray,
    *,
    qcd_label: int = 0,
    hbb_label: int = 1,
    target_misid_rate: float = 0.01,
    eps: float = 1e-12,
) -> Optional[float]:
    """Compute signal (Hbb) efficiency at a cut giving target QCD misidentification rate.

    TXbb = Hbb_score / QCD_score. Find threshold on TXbb where 1% of QCD jets pass,
    then report fraction of Hbb jets passing that threshold.

    Args:
        probs: (N, num_classes) class probabilities
        targets: (N,) ground-truth class indices
        qcd_label: class index for QCD (default 0)
        hbb_label: class index for Hbb (default 1)
        target_misid_rate: target QCD misid rate (default 0.01 = 1%)
        eps: small constant to avoid div-by-zero in TXbb

    Returns:
        efficiency@1% (signal efficiency at 1% QCD misid) or None if insufficient data
    """
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets).reshape(-1).astype(int)
    if probs.ndim != 2 or probs.shape[0] != len(targets):
        return None
    if probs.shape[1] < 2:
        return None

    hbb_prob = probs[:, hbb_label]
    qcd_prob = probs[:, qcd_label] + eps
    txbb = hbb_prob / qcd_prob

    qcd_mask = targets == qcd_label
    hbb_mask = targets == hbb_label
    n_qcd = qcd_mask.sum()
    n_hbb = hbb_mask.sum()
    if n_qcd < 2 or n_hbb < 1:
        return None

    # Percentile: (1 - target_misid_rate) of QCD jets below threshold
    # -> target_misid_rate of QCD jets above threshold (misidentified)
    percentile = (1.0 - target_misid_rate) * 100.0
    threshold = np.percentile(txbb[qcd_mask], percentile)
    efficiency = (txbb[hbb_mask] > threshold).mean()
    return float(efficiency)


def _jsd_from_hists(
    p_counts: np.ndarray, q_counts: np.ndarray, eps: float = 1e-10
) -> float:
    """JSD between two unnormalised histogram count arrays (same as finetune)."""
    p = p_counts / (p_counts.sum() + eps)
    q = q_counts / (q_counts.sum() + eps)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
    kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
    return float(0.5 * kl_pm + 0.5 * kl_qm)


def compute_accuracy_and_jsd(
    preds: np.ndarray,
    targets: np.ndarray,
    jsd_aux: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    n_bins: int = 50,
    probs: Optional[np.ndarray] = None,
    qcd_label: int = 0,
    hbb_label: int = 1,
    target_misid_rate: float = 0.01,
) -> dict:
    """Compute acc, balanced_acc, mean JSD, and efficiency@1%.

    **JSD (mass decorrelation):** uses **true QCD** jets only. For each predicted
    class, builds a mass histogram of jets with (ground truth = QCD, pred = c)
    and compares it to the inclusive true-QCD mass histogram; returns the mean
    JSD over predicted classes (needs ≥2 such bins with support).

    Args:
        preds: (N,) predicted class indices
        targets: (N,) ground-truth class indices
        jsd_aux: (N,) auxiliary variable for JSD (e.g. jet_sdmass). If None, jsd=0.
        num_classes: If None, inferred from targets
        n_bins: Number of histogram bins for JSD (finetune uses 50 bins)
        probs: (N, C) class probabilities for efficiency@1%. If None, eff_at_1pct omitted.
        qcd_label: class index for QCD (default 0)
        hbb_label: class index for Hbb (default 1)
        target_misid_rate: target QCD misid rate for efficiency (default 0.01)

    Returns:
        dict with keys: acc, balanced_acc (if num_classes>1), jsd, eff_at_1pct (if probs)
    """
    preds = np.asarray(preds).reshape(-1).astype(int)
    targets = np.asarray(targets).reshape(-1).astype(int)
    n = len(preds)
    if n == 0:
        return {"acc": 0.0, "balanced_acc": 0.0, "jsd": 0.0}

    acc = (preds == targets).mean()

    num_classes = num_classes or int(targets.max()) + 1
    balanced_acc = None
    if num_classes > 1:
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
        for c in range(num_classes):
            mask = targets == c
            if mask.any():
                class_total[c] = mask.sum()
                class_correct[c] = ((preds == targets) & mask).sum()
        valid = class_total > 0
        if valid.any():
            balanced_acc = (class_correct[valid] / np.maximum(class_total[valid], 1)).mean()

    jsd_val = 0.0
    if jsd_aux is not None and len(jsd_aux) == n:
        jsd_aux = np.asarray(jsd_aux).reshape(-1).astype(np.float64)
        # JSD: true QCD only. For each predicted class, compare mass of (y=QCD, pred=c)
        # to the inclusive true-QCD mass spectrum (reference = all true QCD).
        qcd_mask = targets == qcd_label
        if not np.any(qcd_mask):
            jsd_val = 0.0
        else:
            aux_q = jsd_aux[qcd_mask]
            lo = float(np.min(aux_q))
            hi = float(np.max(aux_q))
            if hi > lo:
                mass_bins = np.linspace(lo, hi, n_bins + 1)
                jsd_hists: dict = {}
                for c in np.unique(preds[qcd_mask]):
                    c = int(c)
                    m = qcd_mask & (preds == c)
                    h, _ = np.histogram(jsd_aux[m], bins=mass_bins)
                    jsd_hists[c] = h.astype(np.float64)
                classes_with_counts = [c for c, h in jsd_hists.items() if h.sum() > 0]
                if len(classes_with_counts) >= 2:
                    h_ref = sum(jsd_hists[c] for c in classes_with_counts)
                    if h_ref.sum() > 0:
                        jsd_val = float(
                            np.mean(
                                [
                                    _jsd_from_hists(jsd_hists[c], h_ref)
                                    for c in classes_with_counts
                                ]
                            )
                        )

    result = {"acc": float(acc)}
    if balanced_acc is not None:
        result["balanced_acc"] = float(balanced_acc)
    result["jsd"] = jsd_val

    if probs is not None:
        eff = compute_efficiency_at_misid_rate(
            probs, targets,
            qcd_label=qcd_label, hbb_label=hbb_label,
            target_misid_rate=target_misid_rate,
        )
        result["eff_at_1pct"] = eff

    return result
