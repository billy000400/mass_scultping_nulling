#!/usr/bin/env python3
"""Compare QCD jet mass histograms at matched signal efficiency (eff@1% bkg).

For each classifier, the operating point is the standard eff@1% definition from
``metrics.compute_efficiency_at_misid_rate``: threshold on TXbb = P(Hbb)/P(QCD)
such that a fixed fraction of QCD (default 1%) lies above the cut. We then
histogram ``jet_sdmass`` (or ``--mass-key``) for those QCD false positives.

By default, picks a CURE+DisCo run and a DisCo-only run from the Pareto
``summary.csv`` files whose reported ``eff_at_1pct`` values are closest, so the
two models operate at comparable signal acceptance. Overlaying the two QCD
mass shapes checks whether mass sculpting (residual mass dependence in the
background) differs when JSD is used as a decorrelation metric.

Usage (from ``mass_perp_classifier/``):

  python -u scripts/compare_qcd_mass_matched_eff.py

  python scripts/compare_qcd_mass_matched_eff.py \\
    --cure-disco-dir runs/cure_disco_pareto/lambda_7 \\
    --disco-dir runs/disco_pareto/lambda_7.0

Requires ``best.pt`` in each run directory (or pass ``--cure-disco-ckpt`` /
``--disco-ckpt``).

Outputs ``--out`` (PNG), by default a PDF with the same stem, and
``<stem>_meta.json`` with thresholds, efficiencies, and JSD between the two
QCD-passing mass histograms (lower ⇒ more similar sculpting).

Options:

- ``--target-eff`` / ``--target-tol``: restrict auto-pairing to a band in
  efficiency space before minimizing |Δeff|.
- Give only one of ``--cure-disco-dir`` / ``--disco-dir`` to match the other
  arm by closest ``eff_at_1pct`` in the opposite summary.
- ``--no-pdf``: skip the PDF.
- By default, adds **pretrained ParT** (``part_og`` head from ``--Part-model-path``, no
  finetune) at the **same QCD mis-id rate** as the other curves; signal eff@1\%bkg
  usually differs. Use ``--no-part-og`` to omit.
- ``--bootstrap B``: optional 16–84% pointwise bands (default **B=0**, off). ``--no-bootstrap``
  forces off. ``--bootstrap-seed`` sets the RNG for B>0.
- ``--bootstrap B`` / ``--bootstrap-seed``: optional 16–84% pointwise bands (B
  resamples of validation rows, with replacement) on densities and
  model/inclusive mass ratios. Reflects **finite val-set** uncertainty on the
  cut and the histograms, with **shared resampling** so bands are
  correlated across curves; not physics MC statistics.

**Performance:** Avoid importing the full finetune module (TensorBoard, etc.). The
main wait before ``Val samples:`` is usually **reading the full validation array**
from HDF5. Use ``--max-val-events 80000`` (stratified subset) for interactive runs;
the loader reads val labels, then either one sequential read of the full val
feature block (if it fits under ``COMPARE_QCD_MAX_FULL_VAL_BYTES``, default 1.2e9)
plus NumPy indexing, or chunked slice reads. Omit ``--max-val-events`` for the
full val set. Use ``python -u`` for line-buffered logs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# First feedback as soon as the interpreter executes this file (before NumPy/Torch).
if sys.argv and "compare_qcd_mass_matched_eff" in os.path.basename(sys.argv[0]):
    print(
        "compare_qcd_mass_matched_eff: loading numpy / torch / src (often a few seconds) …",
        flush=True,
    )

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Run from repo root: mass_perp_classifier/
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import h5py

from src.core.data import (
    join_path,
    load_aux_from_h5,
    load_h5_group,
    make_loader,
    stratified_split,
    has_split,
)
from src.core.metrics import (
    _jsd_from_hists,
    compute_efficiency_at_misid_rate,
)
from src.core.model import PerpClassifier, load_pretrained_part_weights
from src.core.projections import load_svd_basis, project_svd
from src.core.utils import infer_num_classes, set_seed

DEFAULT_H5 = "/scope-vol/hidden_out/train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5"
DEFAULT_PART_MODEL = "/scope-vol/mass_perp_classifier/logs/no_bias/best.pt"


def _load_svd(
    svd_forget: Optional[str],
    svd_retain: Optional[str],
    svd_k: Optional[int],
    svd_variance: float,
    device: str,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Same basis loading as finetune (without importing the finetune module)."""
    if svd_forget is None:
        return None, None, None, None
    Vh_k, mean_k, _ = load_svd_basis(svd_forget, svd_k, svd_variance)
    forget_basis = torch.from_numpy(Vh_k).to(device)
    forget_mean = torch.from_numpy(mean_k).to(device)
    retain_basis = retain_mean = None
    if svd_retain:
        Vh_r, mean_r, _ = load_svd_basis(svd_retain, svd_k, svd_variance)
        retain_basis = torch.from_numpy(Vh_r).to(device)
        retain_mean = torch.from_numpy(mean_r).to(device)
    return forget_basis, forget_mean, retain_basis, retain_mean


def _read_summary_csv(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            row["disco_lambda"] = float(row["disco_lambda"])
            row["eff_at_1pct"] = float(row["eff_at_1pct"])
            if "jsd" in row and row["jsd"]:
                row["jsd"] = float(row["jsd"])
            rows.append(row)
    return rows


def _resolve_lambda_run_dir(pareto_root: str, lam: float) -> Optional[str]:
    """Return run directory whose ``lambda_*`` name is closest to ``lam``."""
    target = float(lam)
    if not os.path.isdir(pareto_root):
        return None
    best: Optional[str] = None
    best_err = float("inf")
    for name in os.listdir(pareto_root):
        if not name.startswith("lambda_"):
            continue
        sub = os.path.join(pareto_root, name)
        if not os.path.isdir(sub):
            continue
        try:
            v = float(name.replace("lambda_", ""))
        except ValueError:
            continue
        err = abs(v - target)
        if err < best_err:
            best_err = err
            best = sub
    return best


def _read_eff_from_run_dir(run_dir: str) -> float:
    path = os.path.join(run_dir, "best_metrics.json")
    with open(path) as f:
        return float(json.load(f)["eff_at_1pct"])


def _read_jsd_from_run_dir(run_dir: str) -> Optional[float]:
    path = os.path.join(run_dir, "best_metrics.json")
    try:
        with open(path) as f:
            j = json.load(f)
        if "jsd" not in j:
            return None
        return float(j["jsd"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def _find_eff_matched_pair(
    cure_rows: List[Dict[str, Any]],
    disco_rows: List[Dict[str, Any]],
    *,
    target_eff: Optional[float] = None,
    target_tol: float = 0.2,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """Pick (cure_row, disco_row) minimizing |eff_c - eff_d|, optionally near ``target_eff``."""
    pairs: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for c in cure_rows:
        for d in disco_rows:
            diff = abs(c["eff_at_1pct"] - d["eff_at_1pct"])
            pairs.append((diff, c, d))
    if target_eff is not None:
        filtered = [
            p
            for p in pairs
            if abs(p[1]["eff_at_1pct"] - target_eff) <= target_tol
            and abs(p[2]["eff_at_1pct"] - target_eff) <= target_tol
        ]
        if filtered:
            pairs = filtered
    pairs.sort(key=lambda x: x[0])
    assert pairs
    return pairs[0][1], pairs[0][2], pairs[0][0]


def _labels_array_from_h5(y_raw: np.ndarray) -> np.ndarray:
    """Match ``load_h5_group`` label decoding."""
    N = len(y_raw)
    per = y_raw.reshape(N, -1)
    if per.shape[1] == 1:
        y_vals = per[:, 0]
        y_np = np.rint(y_vals).astype(np.int64)
        if not np.all(np.isfinite(y_vals)):
            raise ValueError("Non-finite values in labels.")
    else:
        y_np = per.argmax(axis=1).astype(np.int64)
    return y_np


def _stratified_subsample_idx(y: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    """Balanced indices of length ``max_n`` (or fewer if not enough events)."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y).reshape(-1)
    max_n = min(max_n, len(y))
    classes = np.unique(y)
    picks: List[int] = []
    masks = {int(c): np.where(y == c)[0] for c in classes}
    for c in masks:
        rng.shuffle(masks[c])
    ptr = {int(c): 0 for c in classes}
    ci = 0
    clist = [int(c) for c in classes]
    while len(picks) < max_n:
        if all(ptr[c] >= len(masks[c]) for c in clist):
            break
        c = clist[ci % len(clist)]
        ci += 1
        if ptr[c] < len(masks[c]):
            picks.append(int(masks[c][ptr[c]]))
            ptr[c] += 1
    return np.asarray(picks, dtype=np.int64)


def _read_2d_dataset_indexed(ds: h5py.Dataset, idx: np.ndarray) -> np.ndarray:
    """Gather 2D dataset rows by ``idx`` using contiguous slice reads (fast vs. fancy index)."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return np.zeros((0, int(np.prod(ds.shape[1:]))), dtype=np.float32)
    nfeat = int(np.prod(ds.shape[1:]))
    out = np.empty((len(idx), nfeat), dtype=np.float32)
    order = np.argsort(idx, kind="mergesort")
    s = idx[order]
    i = 0
    while i < len(s):
        start_row = int(s[i])
        j = i
        while j + 1 < len(s) and int(s[j + 1]) == int(s[j]) + 1:
            j += 1
        end_row = int(s[j])
        nrows = j - i + 1
        block = np.asarray(ds[start_row : end_row + 1], dtype=np.float32).reshape(nrows, nfeat)
        out[order[i : j + 1]] = block
        i = j + 1
    return out


def _read_1d_dataset_indexed(ds: h5py.Dataset, idx: np.ndarray) -> np.ndarray:
    """Gather 1D dataset rows by ``idx`` using contiguous slice reads."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return np.zeros((0,), dtype=np.float32)
    out = np.empty((len(idx),), dtype=np.float32)
    order = np.argsort(idx, kind="mergesort")
    s = idx[order]
    i = 0
    while i < len(s):
        start_row = int(s[i])
        j = i
        while j + 1 < len(s) and int(s[j + 1]) == int(s[j]) + 1:
            j += 1
        end_row = int(s[j])
        nrows = j - i + 1
        block = np.asarray(ds[start_row : end_row + 1], dtype=np.float32).reshape(nrows)
        out[order[i : j + 1]] = block
        i = j + 1
    return out


def _load_val_indexed(
    h5: str,
    val_group: str,
    x_key: str,
    y_key: str,
    mass_key: str,
    max_events: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read only ``max_events`` stratified rows from ``val_group`` (fast for large H5)."""
    path_y = join_path(val_group, y_key)
    with h5py.File(h5, "r") as f:
        if path_y not in f:
            raise KeyError(path_y)
        print(f"  reading all val labels ({path_y}) …", flush=True)
        y_raw = np.array(f[path_y][:])
        y_full = _labels_array_from_h5(y_raw)
        idx = _stratified_subsample_idx(y_full, max_events, seed)
        x_ds = f[join_path(val_group, x_key)]
        m_ds = f[join_path(val_group, mass_key)]
        n = int(x_ds.shape[0])
        nfeat = int(np.prod(x_ds.shape[1:]))
        row_bytes_x = int(np.dtype(np.float32).itemsize * nfeat)
        est_bytes = n * (row_bytes_x + 4)
        # One sequential read + NumPy fancy index in RAM beats millions of tiny HDF5 reads.
        max_full = int(
            os.environ.get("COMPARE_QCD_MAX_FULL_VAL_BYTES", str(1_200_000_000))
        )
        if est_bytes <= max_full:
            print(
                f"  loading full val features ({n:,}×{nfeat}, ~{est_bytes / 1e9:.2f} GB) then subsampling …",
                flush=True,
            )
            x_all = np.asarray(x_ds[:], dtype=np.float32).reshape(n, nfeat)
            m_all = np.asarray(m_ds[:], dtype=np.float32).reshape(-1)
            x_va = x_all[idx]
            mass_va = m_all[idx]
        else:
            print(
                f"  stratified idx: {len(idx):,} rows — loading features via contiguous HDF5 slices …",
                flush=True,
            )
            x_va = _read_2d_dataset_indexed(x_ds, idx)
            mass_va = _read_1d_dataset_indexed(m_ds, idx)
    y_va = y_full[idx]
    return x_va, y_va, mass_va


def _load_val_split(
    h5: str,
    train_group: str,
    val_group: str,
    x_key: str,
    y_key: str,
    mass_key: str,
    train_frac: float,
    data_fraction: float,
    seed: int,
    *,
    max_val_events: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    split = has_split(h5, train_group, val_group, x_key)
    if split and max_val_events is not None and max_val_events > 0:
        x_va, y_va, mass_va = _load_val_indexed(
            h5, val_group, x_key, y_key, mass_key, max_val_events, seed
        )
        return x_va, y_va, mass_va, split
    if split:
        x_va, y_va = load_h5_group(h5, val_group, x_key, y_key, data_fraction=data_fraction)
        mass_va = load_aux_from_h5(h5, val_group, mass_key, len(x_va))
    else:
        x_all, y_all = load_h5_group(h5, "", x_key, y_key, data_fraction=data_fraction)
        train_idx, val_idx = stratified_split(y_all, train_frac, seed)
        x_va = x_all[val_idx]
        y_va = y_all[val_idx]
        mass_all = load_aux_from_h5(h5, "", mass_key, len(x_all))
        mass_va = mass_all[val_idx]
        if max_val_events is not None and max_val_events > 0 and len(x_va) > max_val_events:
            sub = _stratified_subsample_idx(y_va, max_val_events, seed)
            x_va, y_va, mass_va = x_va[sub], y_va[sub], mass_va[sub]
    return x_va, y_va, mass_va, split


def _build_model(
    experiment: str,
    in_dim: int,
    num_classes: int,
    device: str,
    part_path: str,
) -> PerpClassifier:
    use_pretrained = experiment in ("part_og", "rav", "cure", "cure_disco") and bool(
        part_path
    )
    pretrained_state = None
    if use_pretrained:
        pretrained_state = load_pretrained_part_weights(
            part_path, in_dim=in_dim, num_classes=num_classes, device=device
        )
    model = PerpClassifier(
        in_dim=in_dim,
        num_classes=num_classes,
        for_inference=True,
        use_pretrained_norm_fc=use_pretrained,
        pretrained_state_dict=pretrained_state,
        input_is_post_ln=True,
    )
    model.to(device)
    return model


@torch.no_grad()
def _predict_probs(
    model: PerpClassifier,
    x_va: np.ndarray,
    y_va: np.ndarray,
    mass_va: np.ndarray,
    batch_size: int,
    device: str,
    svd_f,
    svd_fm,
    svd_r,
    svd_rm,
    alpha: float,
    *,
    desc: str = "infer",
) -> np.ndarray:
    loader = make_loader(
        x_va,
        y_va,
        batch_size=batch_size,
        shuffle=False,
        jsd_aux_np=mass_va,
    )
    outs = []
    for batch in tqdm(loader, desc=desc, leave=False):
        x = batch[0].to(device)
        if svd_f is not None:
            x = project_svd(x, svd_f, svd_fm, svd_r, svd_rm, alpha=alpha)
        probs = model(x)
        if isinstance(probs, (tuple, list)):
            probs = probs[0]
        outs.append(probs.cpu().numpy())
    return np.concatenate(outs, axis=0)


def _txbb(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # HEP convention: TXbb = P(Hbb) / (P(Hbb) + P(QCD)).
    # Percentile-based eff@misid is invariant to monotone transforms, so any
    # previously cached eff@1% values remain valid; only the raw TXbb scalar
    # range changes (was P(Hbb)/P(QCD) -- unbounded -- now bounded in [0, 1]).
    return probs[:, 1] / (probs[:, 0] + probs[:, 1] + eps)


def _load_ckpt_state(path: str, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _mass_bin_edges(lo: float, hi: float, width: float) -> np.ndarray:
    """Uniform bin edges of given ``width`` covering ``[lo, hi]``."""
    if width <= 0:
        raise SystemExit("--mass-bin-width must be positive")
    e0 = np.floor(lo / width) * width
    e1 = np.ceil(hi / width) * width
    edges = np.arange(e0, e1 + 0.5 * width, width, dtype=np.float64)
    if len(edges) < 2:
        edges = np.array([lo, hi], dtype=np.float64)
    return edges


def _qcd_passing_masses(
    probs: np.ndarray,
    masses: np.ndarray,
    targets: np.ndarray,
    *,
    qcd_label: int = 0,
    misid_rate: float = 0.01,
) -> Tuple[np.ndarray, float, float]:
    """QCD jets above eff@1% threshold; returns masses, threshold, eff_recomputed."""
    tx = _txbb(probs)
    qcd_mask = targets == qcd_label
    n_qcd = int(qcd_mask.sum())
    if n_qcd < 2:
        raise SystemExit(
            f"Need at least 2 QCD jets in validation for eff@1% (got {n_qcd}). "
            "The H5 may list one class before the other; "
            "--data-fraction < 1 takes only the first N rows and can drop an entire class. "
            "Use --data-fraction 1.0 or a larger fraction."
        )
    percentile = (1.0 - misid_rate) * 100.0
    thr = float(np.percentile(tx[qcd_mask], percentile))
    passing = qcd_mask & (tx > thr)
    eff_r = compute_efficiency_at_misid_rate(
        probs, targets, qcd_label=0, hbb_label=1, target_misid_rate=misid_rate
    )
    return masses[passing], thr, float(eff_r) if eff_r is not None else float("nan")


def _passing_qcd_masses_from_sample(
    probs: np.ndarray,
    masses: np.ndarray,
    targets: np.ndarray,
    misid_rate: float,
    qcd_label: int = 0,
) -> np.ndarray:
    """Masses of QCD jets passing the eff@1% mis-id cut; empty if <2 QCD in sample."""
    tx = _txbb(probs)
    qcd_mask = targets == qcd_label
    if int(qcd_mask.sum()) < 2:
        return np.array([], dtype=np.float64)
    percentile = (1.0 - misid_rate) * 100.0
    thr = float(np.percentile(tx[qcd_mask], percentile))
    passing = qcd_mask & (tx > thr)
    return masses[passing]


def _bootstrap_mass_uncertainty(
    probs_c: np.ndarray,
    probs_d: np.ndarray,
    mass_va: np.ndarray,
    y_va: np.ndarray,
    bins: np.ndarray,
    *,
    misid_rate: float,
    n_boot: int,
    seed: int,
    probs_part_og: Optional[np.ndarray] = None,
) -> dict:
    """Nonparametric bootstrap on validation rows (with replacement): shared resampling for
    inclusive QCD + both models' passing sets. Returns 16/50/84% binwise percentiles for densities
    and ratios (approximately 1σ for Gaussian-like bootstrap distributions).

    This captures **finite val statistics** (including correlation between cut and shape); it does
    **not** include train/inference MC statistical uncertainty unless you re-run models.
    """
    rng = np.random.RandomState(int(seed))
    N = len(y_va)
    n_bins = len(bins) - 1
    H_inc = np.zeros((n_boot, n_bins), dtype=np.float64)
    H_c = np.zeros((n_boot, n_bins), dtype=np.float64)
    H_d = np.zeros((n_boot, n_bins), dtype=np.float64)
    has_p = probs_part_og is not None
    H_p = (
        np.zeros((n_boot, n_bins), dtype=np.float64) if has_p else None
    )
    for b in range(n_boot):
        idx = rng.randint(0, N, size=N)
        yb = y_va[idx]
        mb = mass_va[idx]
        pc = probs_c[idx]
        pd = probs_d[idx]
        m_inc = mb[yb == 0]
        if m_inc.size > 0:
            h, _ = np.histogram(m_inc, bins=bins, density=True)
            H_inc[b] = h
        m_pc = _passing_qcd_masses_from_sample(pc, mb, yb, misid_rate)
        m_pd = _passing_qcd_masses_from_sample(pd, mb, yb, misid_rate)
        if m_pc.size > 0:
            h, _ = np.histogram(m_pc, bins=bins, density=True)
            H_c[b] = h
        if m_pd.size > 0:
            h, _ = np.histogram(m_pd, bins=bins, density=True)
            H_d[b] = h
        if has_p and H_p is not None:
            pp = probs_part_og[idx]
            m_pp = _passing_qcd_masses_from_sample(pp, mb, yb, misid_rate)
            if m_pp.size > 0:
                h, _ = np.histogram(m_pp, bins=bins, density=True)
                H_p[b] = h
    p_lo, p_mid, p_hi = 16.0, 50.0, 84.0
    inc_pct = np.percentile(H_inc, [p_lo, p_mid, p_hi], axis=0)
    c_pct = np.percentile(H_c, [p_lo, p_mid, p_hi], axis=0)
    d_pct = np.percentile(H_d, [p_lo, p_mid, p_hi], axis=0)
    eps = 1e-15
    R_c = H_c / (H_inc + eps)
    R_d = H_d / (H_inc + eps)
    r_c_pct = np.nanpercentile(R_c, [p_lo, p_mid, p_hi], axis=0)
    r_d_pct = np.nanpercentile(R_d, [p_lo, p_mid, p_hi], axis=0)
    out: Dict[str, Any] = {
        "inclusive_density": inc_pct,
        "cure_disco_density": c_pct,
        "disco_density": d_pct,
        "ratio_cure": r_c_pct,
        "ratio_disco": r_d_pct,
        "percentile_levels": (p_lo, p_mid, p_hi),
    }
    if has_p and H_p is not None:
        p_pct = np.percentile(H_p, [p_lo, p_mid, p_hi], axis=0)
        R_p = H_p / (H_inc + eps)
        r_p_pct = np.nanpercentile(R_p, [p_lo, p_mid, p_hi], axis=0)
        out["part_og_density"] = p_pct
        out["ratio_part_og"] = r_p_pct
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--train-group", default="/train")
    ap.add_argument("--val-group", default="/val")
    ap.add_argument("--x-key", default="cls_tokens_ln")
    ap.add_argument("--y-key", default="label")
    ap.add_argument("--mass-key", default="jet_sdmass", help="Mass variable in H5")
    ap.add_argument(
        "--mass-bin-width",
        type=float,
        default=10.0,
        help="Fixed width of mass histogram bins (same units as --mass-key; default: 10)",
    )
    ap.add_argument(
        "--mass-plot-min",
        type=float,
        default=75.0,
        help="Lower edge of mass axis and histogram range (default: 75, matches H5 window)",
    )
    ap.add_argument(
        "--mass-plot-max",
        type=float,
        default=175.0,
        help="Upper edge of mass axis and histogram range (default: 175, matches H5 window)",
    )
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument(
        "--data-fraction",
        type=float,
        default=1.0,
        help="Fraction of rows read from each H5 split (first N rows only; can remove an entire "
        "class if the file is sorted by label). Prefer 1.0 for real plots.",
    )
    ap.add_argument(
        "--max-val-events",
        type=int,
        default=None,
        metavar="N",
        help="If the H5 has /train and /val, load at most N validation jets with stratified class "
        "sampling: reads all val labels (small), then only those rows of features/mass — avoids "
        "multi‑minute full-array I/O. Omit for the full val set (slow on large files). "
        "Efficiencies will not match best_metrics.json exactly when N is finite.",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--misid-rate", type=float, default=0.01)
    ap.add_argument("--Part-model-path", default=DEFAULT_PART_MODEL, dest="part_path")
    ap.add_argument(
        "--no-part-og",
        action="store_true",
        help="Skip pretrained ParT (part_og) baseline: LayerNorm+head from --Part-model-path, no finetune.",
    )
    ap.add_argument(
        "--svd-forget",
        default="/scope-vol/svd_results/QCD_res_cls_tokens_ln_svd.npz",
        help="SVD forget basis for cure_disco (must match training)",
    )
    ap.add_argument("--svd-retain", default=None)
    ap.add_argument("--svd-k", type=int, default=None)
    ap.add_argument("--svd-variance", type=float, default=0.95)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument(
        "--cure-summary",
        default=os.path.join(_PROJ, "runs/cure_disco_pareto/summary.csv"),
    )
    ap.add_argument(
        "--disco-summary",
        default=os.path.join(_PROJ, "runs/disco_pareto/summary.csv"),
    )
    ap.add_argument("--cure-disco-dir", default=None, help="Run dir with best.pt (cure_disco)")
    ap.add_argument("--disco-dir", default=None, help="Run dir with best.pt (disco)")
    ap.add_argument("--cure-disco-ckpt", default=None)
    ap.add_argument("--disco-ckpt", default=None)
    ap.add_argument(
        "--out",
        default=os.path.join(_PROJ, "runs/qcd_mass_matched_eff.png"),
    )
    ap.add_argument(
        "--out-pdf",
        default=None,
        help="PDF path (default: same stem as --out with .pdf)",
    )
    ap.add_argument("--no-pdf", action="store_true", help="Do not write a PDF copy")
    ap.add_argument(
        "--target-eff",
        type=float,
        default=None,
        help="If set, prefer Pareto pairs whose eff@1%% are both within --target-tol of this value",
    )
    ap.add_argument(
        "--target-tol",
        type=float,
        default=0.2,
        help="Tolerance for --target-eff (default: 0.2)",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=0,
        metavar="B",
        help="Default 0 = **disabled** (no uncertainty bands). If B>0, draw B nonparametric "
        "bootstrap resamples of the validation set (with replacement) and add 16–84%% pointwise "
        "bands on the density and ratio panels. One resample is shared for inclusive QCD and all "
        "classifiers. To turn off after enabling: pass --bootstrap 0.",
    )
    ap.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Seed for --bootstrap (default: 0).",
    )
    ap.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Force bootstrap off (--bootstrap 0). Overrides a positive --bootstrap if both are given.",
    )
    args = ap.parse_args()
    if args.no_bootstrap:
        args.bootstrap = 0
    if args.mass_plot_min >= args.mass_plot_max:
        raise SystemExit("--mass-plot-min must be less than --mass-plot-max")

    print(
        "compare_qcd_mass_matched_eff: starting (imports done; resolving runs …)",
        flush=True,
    )
    device = args.device
    if isinstance(device, str) and device.lower().startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(
                "You passed --device cuda but torch.cuda.is_available() is False. "
                "Use --device cpu or fix the GPU environment."
            )
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print(f"Device: {device}", flush=True)

    set_seed(args.seed)

    cure_dir = args.cure_disco_dir
    disco_dir = args.disco_dir
    cure_row: Optional[Dict[str, Any]] = None
    disco_row: Optional[Dict[str, Any]] = None
    root_cure = os.path.dirname(args.cure_summary)
    root_disco = os.path.dirname(args.disco_summary)

    if cure_dir is None or disco_dir is None:
        cure_rows = _read_summary_csv(args.cure_summary)
        disco_rows = _read_summary_csv(args.disco_summary)
        if cure_dir is None and disco_dir is None:
            cure_row, disco_row, eff_diff = _find_eff_matched_pair(
                cure_rows,
                disco_rows,
                target_eff=args.target_eff,
                target_tol=args.target_tol,
            )
            cure_dir = _resolve_lambda_run_dir(root_cure, cure_row["disco_lambda"])
            disco_dir = _resolve_lambda_run_dir(root_disco, disco_row["disco_lambda"])
            print(
                f"Auto-matched pair (|Δeff@1%| = {eff_diff:.4f}):\n"
                f"  CURE+DisCo λ={cure_row['disco_lambda']} eff={cure_row['eff_at_1pct']:.4f} "
                f"JSD={cure_row.get('jsd', 'n/a')}\n"
                f"  DisCo only   λ={disco_row['disco_lambda']} eff={disco_row['eff_at_1pct']:.4f} "
                f"JSD={disco_row.get('jsd', 'n/a')}"
            )
        elif cure_dir is None:
            eff_d = _read_eff_from_run_dir(disco_dir)
            cure_row = min(
                cure_rows, key=lambda r: abs(r["eff_at_1pct"] - eff_d)
            )
            cure_dir = _resolve_lambda_run_dir(root_cure, cure_row["disco_lambda"])
            print(
                f"DisCo (user dir): eff@1% (best_metrics)={eff_d:.4f}\n"
                f"Matched CURE+DisCo: λ={cure_row['disco_lambda']} "
                f"eff={cure_row['eff_at_1pct']:.4f} -> {cure_dir}"
            )
        else:
            eff_c = _read_eff_from_run_dir(cure_dir)
            disco_row = min(
                disco_rows, key=lambda r: abs(r["eff_at_1pct"] - eff_c)
            )
            disco_dir = _resolve_lambda_run_dir(root_disco, disco_row["disco_lambda"])
            print(
                f"CURE+DisCo (user dir): eff@1% (best_metrics)={eff_c:.4f}\n"
                f"Matched DisCo: λ={disco_row['disco_lambda']} "
                f"eff={disco_row['eff_at_1pct']:.4f} -> {disco_dir}"
            )
    else:
        print("Using user-provided run dirs (both checkpoints).")

    if not cure_dir or not os.path.isdir(cure_dir):
        raise SystemExit(f"Invalid --cure-disco-dir: {cure_dir!r}")
    if not disco_dir or not os.path.isdir(disco_dir):
        raise SystemExit(f"Invalid --disco-dir: {disco_dir!r}")

    ckpt_cure = args.cure_disco_ckpt or os.path.join(cure_dir, "best.pt")
    ckpt_disco = args.disco_ckpt or os.path.join(disco_dir, "best.pt")
    for label, path in [("CURE+DisCo", ckpt_cure), ("DisCo", ckpt_disco)]:
        if not os.path.isfile(path):
            raise SystemExit(
                f"Missing checkpoint for {label}: {path}\n"
                "Train the Pareto runs with finetune (saves best.pt) or pass "
                "--cure-disco-ckpt / --disco-ckpt."
            )
    print(
        f"Checkpoints OK:\n  CURE+DisCo: {ckpt_cure}\n  DisCo:      {ckpt_disco}",
        flush=True,
    )

    print(
        "Loading validation split from H5 …\n"
        f"  {args.h5}\n"
        f"  val_group={args.val_group!r}  data_fraction={args.data_fraction}  "
        f"max_val_events={args.max_val_events!r}",
        flush=True,
    )
    if args.max_val_events is None:
        print(
            "  (Full val load can take many minutes on large files. For a fast stratified subset, "
            "pass e.g. --max-val-events 80000.)",
            flush=True,
        )
    t_h5 = time.perf_counter()
    x_va, y_va, mass_va, _split = _load_val_split(
        args.h5,
        args.train_group,
        args.val_group,
        args.x_key,
        args.y_key,
        args.mass_key,
        args.train_frac,
        args.data_fraction,
        args.seed,
        max_val_events=args.max_val_events,
    )
    if y_va is None:
        raise SystemExit("Labels required in H5 for eff@1% and QCD mask.")

    in_dim = int(x_va.shape[1])
    num_classes = infer_num_classes(y_va)
    print(
        f"Val samples: {len(x_va):,} | H5 load: {time.perf_counter() - t_h5:.1f}s | "
        f"device={device} | batch_size={args.batch_size}",
        flush=True,
    )
    if args.max_val_events is not None:
        print(
            "Note: subsampled validation — recomputed eff@1%% and sculpting shapes are approximate.",
            flush=True,
        )
    if str(device).lower() == "cpu":
        print(
            "Note: CPU inference over the val set twice can take many minutes. "
            "Use --device cuda if available, or reduce --max-val-events.",
            flush=True,
        )

    svd_f, svd_fm, svd_r, svd_rm = _load_svd(
        args.svd_forget,
        args.svd_retain,
        args.svd_k,
        args.svd_variance,
        device,
    )

    model_c = _build_model("cure_disco", in_dim, num_classes, device, args.part_path)
    model_c.load_state_dict(_load_ckpt_state(ckpt_cure, device))
    model_c.eval()
    t0 = time.perf_counter()
    probs_c = _predict_probs(
        model_c, x_va, y_va, mass_va, args.batch_size, device,
        svd_f, svd_fm, svd_r, svd_rm, args.alpha,
        desc="CURE+DisCo",
    )
    print(f"CURE+DisCo inference: {time.perf_counter() - t0:.1f}s", flush=True)

    model_d = _build_model("disco", in_dim, num_classes, device, args.part_path)
    model_d.load_state_dict(_load_ckpt_state(ckpt_disco, device))
    model_d.eval()
    t0 = time.perf_counter()
    probs_d = _predict_probs(
        model_d, x_va, y_va, mass_va, args.batch_size, device,
        None, None, None, None, args.alpha,
        desc="DisCo",
    )
    print(f"DisCo inference: {time.perf_counter() - t0:.1f}s", flush=True)

    include_part_og = (not args.no_part_og) and bool(args.part_path) and os.path.isfile(
        str(args.part_path)
    )
    if not include_part_og and not args.no_part_og:
        print(
            f"Warning: ParT baseline skipped (missing --Part-model-path file: {args.part_path!r}).",
            flush=True,
        )
    probs_p: Optional[np.ndarray] = None
    if include_part_og:
        model_p = _build_model("part_og", in_dim, num_classes, device, args.part_path)
        model_p.eval()
        t0 = time.perf_counter()
        probs_p = _predict_probs(
            model_p,
            x_va,
            y_va,
            mass_va,
            args.batch_size,
            device,
            None,
            None,
            None,
            None,
            args.alpha,
            desc="ParT pretrained",
        )
        print(f"ParT pretrained (no finetune) inference: {time.perf_counter() - t0:.1f}s", flush=True)

    m_c, thr_c, eff_c = _qcd_passing_masses(
        probs_c, mass_va, y_va, misid_rate=args.misid_rate
    )
    m_d, thr_d, eff_d = _qcd_passing_masses(
        probs_d, mass_va, y_va, misid_rate=args.misid_rate
    )
    if include_part_og and probs_p is not None:
        m_p, thr_p, eff_p = _qcd_passing_masses(
            probs_p, mass_va, y_va, misid_rate=args.misid_rate
        )
    else:
        m_p = np.array([], dtype=np.float64)
        thr_p = float("nan")
        eff_p = float("nan")

    qcd_mask = y_va == 0
    m_inclusive = mass_va[qcd_mask]

    pm_lo, pm_hi = float(args.mass_plot_min), float(args.mass_plot_max)
    d_lo = float(min(mass_va.min(), m_inclusive.min()))
    d_hi = float(max(mass_va.max(), m_inclusive.max()))
    if d_lo < pm_lo or d_hi > pm_hi:
        print(
            f"Warning: some masses lie outside [{pm_lo}, {pm_hi}] (data span ~{d_lo:.1f}–{d_hi:.1f}); "
            "np.histogram omits those events from the plotted bins.",
            flush=True,
        )
    bins = _mass_bin_edges(pm_lo, pm_hi, args.mass_bin_width)

    h_inc, _ = np.histogram(m_inclusive, bins=bins, density=True)
    h_c, _ = np.histogram(m_c, bins=bins, density=True)
    h_d, _ = np.histogram(m_d, bins=bins, density=True)
    h_p, _ = np.histogram(m_p, bins=bins, density=True) if len(m_p) > 0 else (
        np.zeros(len(bins) - 1),
        None,
    )
    h_inc_counts, _ = np.histogram(m_inclusive, bins=bins)
    h_c_counts, _ = np.histogram(m_c, bins=bins)
    h_d_counts, _ = np.histogram(m_d, bins=bins)
    h_p_counts, _ = np.histogram(m_p, bins=bins) if len(m_p) > 0 else (
        np.zeros(len(bins) - 1),
        None,
    )
    if len(m_c) == 0 or len(m_d) == 0:
        jsd_shapes = float("nan")
        jsd_sculpt_c = jsd_sculpt_d = float("nan")
        print("Warning: empty QCD passing set for one model; JSD between shapes undefined.")
    else:
        jsd_shapes = _jsd_from_hists(
            h_c_counts.astype(np.float64), h_d_counts.astype(np.float64)
        )
        # Mass sculpting: how different passing-QCD mass shape is from inclusive QCD (same bins).
        jsd_sculpt_c = _jsd_from_hists(
            h_c_counts.astype(np.float64), h_inc_counts.astype(np.float64)
        )
        jsd_sculpt_d = _jsd_from_hists(
            h_d_counts.astype(np.float64), h_inc_counts.astype(np.float64)
        )
    if len(m_p) > 0 and float(h_p_counts.sum()) > 0:
        jsd_sculpt_p = _jsd_from_hists(
            h_p_counts.astype(np.float64), h_inc_counts.astype(np.float64)
        )
    else:
        jsd_sculpt_p = float("nan")

    centers = 0.5 * (bins[:-1] + bins[1:])

    boot_unc: Optional[dict] = None
    if int(args.bootstrap) > 0:
        t0b = time.perf_counter()
        print(
            f"Bootstrap: {int(args.bootstrap)} val resamples (16–84% density/ratio bands) …",
            flush=True,
        )
        boot_unc = _bootstrap_mass_uncertainty(
            probs_c,
            probs_d,
            mass_va,
            y_va,
            bins,
            misid_rate=args.misid_rate,
            n_boot=int(args.bootstrap),
            seed=int(args.bootstrap_seed),
            probs_part_og=probs_p if include_part_og else None,
        )
        print(f"Bootstrap done in {time.perf_counter() - t0b:.1f}s", flush=True)

    print(
        f"\nValidation QCD count: {qcd_mask.sum()} | "
        f"Passing QCD: CURE+DisCo {len(m_c)}, DisCo {len(m_d)}"
        + (f", ParT(pre) {len(m_p)}" if include_part_og else "")
        + f" (target fraction ≈ {args.misid_rate:.3f})"
    )
    print(f"TXbb thresholds: CURE+DisCo {thr_c:.6g} | DisCo {thr_d:.6g}", end="")
    if include_part_og:
        print(f" | ParT(pre) {thr_p:.6g}", end="")
    print()
    print(f"Recomputed eff@1%bkg: CURE+DisCo {eff_c:.4f} | DisCo {eff_d:.4f}", end="")
    if include_part_og:
        print(
            f" | ParT(pre) {eff_p:.4f} (same QCD mis-id rate; signal eff usually differs from matched pair)",
            end="",
        )
    print()
    jsd_train_c = _read_jsd_from_run_dir(cure_dir)
    jsd_train_d = _read_jsd_from_run_dir(disco_dir)
    if jsd_train_c is not None or jsd_train_d is not None:
        print(
            "Training-metric JSD (best_metrics; true QCD, pred-vs-mass, not the sculpting JSD above): "
            f"CURE+DisCo={jsd_train_c} | DisCo={jsd_train_d}"
        )
    print(
        "QCD mass sculpting JSD( passing QCD || inclusive QCD ), same bins — lower ⇒ closer to inclusive:"
    )
    print(
        f"  CURE+DisCo: {jsd_sculpt_c:.5f}  |  DisCo: {jsd_sculpt_d:.5f}"
        + (f"  |  ParT(pre): {jsd_sculpt_p:.5f}" if include_part_og else "")
    )
    print(
        f"JSD between the two passing-QCD mass shapes only (often tiny when both sculpt similarly): "
        f"{jsd_shapes:.5f}"
    )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    if boot_unc is not None:
        i_lo, _, i_hi = boot_unc["inclusive_density"]
        c_lo, _, c_hi = boot_unc["cure_disco_density"]
        d_lo, _, d_hi = boot_unc["disco_density"]
        _fb_kw = {"step": "mid"}
        ax.fill_between(
            centers, i_lo, i_hi, color="0.5", alpha=0.22, linewidth=0, **_fb_kw
        )
        ax.fill_between(
            centers, c_lo, c_hi, color="C0", alpha=0.25, linewidth=0, **_fb_kw
        )
        ax.fill_between(
            centers, d_lo, d_hi, color="C1", alpha=0.25, linewidth=0, **_fb_kw
        )
        if boot_unc is not None and "part_og_density" in boot_unc:
            p_lo, _, p_hi = boot_unc["part_og_density"]
            ax.fill_between(
                centers, p_lo, p_hi, color="C2", alpha=0.22, linewidth=0, **_fb_kw
            )
    ax.step(centers, h_inc, where="mid", color="0.5", linewidth=1.5, label="All val QCD (norm.)")
    ax.step(centers, h_c, where="mid", color="C0", linewidth=2, label=f"CURE+DisCo QCD @ {args.misid_rate:.0%} mis-id")
    ax.step(centers, h_d, where="mid", color="C1", linewidth=2, label=f"DisCo QCD @ {args.misid_rate:.0%} mis-id")
    if include_part_og and len(m_p) > 0:
        ax.step(
            centers,
            h_p,
            where="mid",
            color="C2",
            linewidth=2,
            linestyle="--",
            label=f"ParT pretrained QCD @ {args.misid_rate:.0%} mis-id",
        )
    ax.set_ylabel("Normalized density")
    ax.set_title(
        "QCD mass sculpting at matched eff@1%bkg (shape vs inclusive QCD; not the training JSD scalar)"
    )
    annot = (
        f"Sculpt JSD vs incl. QCD (↓ better):\n"
        f"CURE+DisCo {jsd_sculpt_c:.4f}\n"
        f"DisCo      {jsd_sculpt_d:.4f}"
        + (f"\nParT(pre)  {jsd_sculpt_p:.4f}" if include_part_og else "")
    )
    if jsd_train_c is not None and jsd_train_d is not None:
        annot += f"\nTrain JSD (json):\n{jsd_train_c:.4f} vs {jsd_train_d:.4f}"
    if boot_unc is not None:
        annot += f"\n16–84%: val bootstrap (B={int(args.bootstrap)})"
    ax.text(
        0.02,
        0.98,
        annot,
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        linespacing=1.15,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.35),
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(pm_lo, pm_hi)

    ax2 = axes[1]
    ratio_c = np.divide(h_c, h_inc + 1e-12)
    ratio_d = np.divide(h_d, h_inc + 1e-12)
    if boot_unc is not None:
        rc_lo, _, rc_hi = boot_unc["ratio_cure"]
        rd_lo, _, rd_hi = boot_unc["ratio_disco"]
        _fb_kw2 = {"step": "mid"}
        ax2.fill_between(
            centers, rc_lo, rc_hi, color="C0", alpha=0.22, linewidth=0, **_fb_kw2
        )
        ax2.fill_between(
            centers, rd_lo, rd_hi, color="C1", alpha=0.22, linewidth=0, **_fb_kw2
        )
        if boot_unc is not None and "ratio_part_og" in boot_unc:
            rp_lo, _, rp_hi = boot_unc["ratio_part_og"]
            ax2.fill_between(
                centers, rp_lo, rp_hi, color="C2", alpha=0.2, linewidth=0, **_fb_kw2
            )
    ax2.axhline(1.0, color="0.5", linestyle="--", linewidth=1)
    ax2.step(centers, ratio_c, where="mid", color="C0", label="CURE+DisCo / inclusive QCD")
    ax2.step(centers, ratio_d, where="mid", color="C1", label="DisCo / inclusive QCD")
    if include_part_og and len(m_p) > 0:
        ratio_p = np.divide(h_p, h_inc + 1e-12)
        ax2.step(
            centers,
            ratio_p,
            where="mid",
            color="C2",
            linestyle="--",
            label="ParT pretrained / inclusive QCD",
        )
    ax2.set_xlabel(args.mass_key)
    ax2.set_ylabel("Ratio to inclusive QCD")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(pm_lo, pm_hi)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    out_pdf_path = None
    if not args.no_pdf:
        out_pdf_path = args.out_pdf or (os.path.splitext(args.out)[0] + ".pdf")

    meta = {
        "max_val_events": args.max_val_events,
        "mass_bin_width": args.mass_bin_width,
        "mass_plot_min": args.mass_plot_min,
        "mass_plot_max": args.mass_plot_max,
        "cure_disco_dir": os.path.abspath(cure_dir),
        "disco_dir": os.path.abspath(disco_dir),
        "cure_disco_ckpt": os.path.abspath(ckpt_cure),
        "disco_ckpt": os.path.abspath(ckpt_disco),
        "misid_rate": args.misid_rate,
        "eff_recomputed_cure_disco": eff_c,
        "eff_recomputed_disco": eff_d,
        "threshold_txbb_cure_disco": thr_c,
        "threshold_txbb_disco": thr_d,
        "n_qcd_passing_cure_disco": int(len(m_c)),
        "n_qcd_passing_disco": int(len(m_d)),
        "jsd_between_passing_mass_hists": jsd_shapes,
        "jsd_qcd_mass_sculpt_vs_inclusive_cure_disco": jsd_sculpt_c,
        "jsd_qcd_mass_sculpt_vs_inclusive_disco": jsd_sculpt_d,
        "training_metric_jsd_best_metrics_cure_disco": jsd_train_c,
        "training_metric_jsd_best_metrics_disco": jsd_train_d,
        "part_og_included": include_part_og,
        "part_model_path": os.path.abspath(str(args.part_path)) if include_part_og else None,
        "eff_recomputed_part_og_pretrained": eff_p if include_part_og else None,
        "threshold_txbb_part_og_pretrained": thr_p if include_part_og else None,
        "n_qcd_passing_part_og_pretrained": int(len(m_p)) if include_part_og else None,
        "jsd_qcd_mass_sculpt_vs_inclusive_part_og_pretrained": jsd_sculpt_p
        if include_part_og
        else None,
    }
    if boot_unc is not None:
        p_lo, p_mid, p_hi = boot_unc["percentile_levels"]
        meta["bootstrap_n"] = int(args.bootstrap)
        meta["bootstrap_seed"] = int(args.bootstrap_seed)
        meta["bootstrap_percentile_levels"] = [float(p_lo), float(p_mid), float(p_hi)]
        meta["bootstrap_interpretation"] = (
            "Nonparametric resample of validation rows (with replacement), same index set for "
            "inclusive QCD, CURE+DisCo, and DisCo. TXbb eff@1% threshold and per-bin densities "
            "recomputed on each replicate. Pointwise 16/84% bands. Does not include separate "
            "uncertainty on learned model parameters or on simulation (MC) statistics."
        )
    if out_pdf_path:
        meta["out_pdf"] = os.path.abspath(out_pdf_path)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved {args.out}")
    if out_pdf_path:
        fig.savefig(out_pdf_path, bbox_inches="tight")
        print(f"Saved {out_pdf_path}")

    if cure_row is not None:
        meta["summary_cure_disco"] = {
            k: cure_row[k]
            for k in ("disco_lambda", "eff_at_1pct", "jsd")
            if k in cure_row
        }
    if disco_row is not None:
        meta["summary_disco"] = {
            k: disco_row[k]
            for k in ("disco_lambda", "eff_at_1pct", "jsd")
            if k in disco_row
        }
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {meta_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
