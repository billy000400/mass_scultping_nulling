#!/usr/bin/env python3
"""Re-evaluate finetuned sweeps with the corrected threshold-based JSD metric.

Walks the given sweep directories (default: a fixed list of canonical Pareto
sweeps), loads each ``lambda_*/best.pt`` checkpoint, runs validation inference
on the same stratified split used by finetune.py (seed=1337, train_frac=0.8),
and writes the corrected metrics into ``best_metrics.json``:

    "jsd"               — NEW: JSD(QCD passing 1% mis-id cut, inclusive QCD)
    "jsd_argmax_legacy" — old per-class-argmax JSD (preserved for comparison)
    "eff_at_1pct"       — recomputed; also kept as cross-check
    "auc"               — preserved if present

The original file is backed up as ``best_metrics.json.argmax_jsd_backup``
(skipped if the backup already exists). After all lambdas in a sweep have been
re-evaluated, ``summary.csv`` is rewritten in disco_lambda order.

Usage:
  cd /scope-vol/mass-sculpting-nulling
  python scripts/recompute_jsd_at_misid.py
  python scripts/recompute_jsd_at_misid.py --sweep-dir runs/disco_pareto_pretrained
  python scripts/recompute_jsd_at_misid.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Resolve project root and import.
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)

from src.core.data import load_h5_group, load_aux_from_h5, stratified_split  # noqa: E402
from src.core.metrics import (  # noqa: E402
    compute_accuracy_and_jsd,
    compute_jsd_argmax,
)
from src.core.model import PerpClassifier  # noqa: E402
from src.core.projections import load_svd_basis  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("recompute_jsd")

DEFAULT_H5 = "/scope-vol/hidden_out/train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5"
DEFAULT_SEED = 1337
DEFAULT_TRAIN_FRAC = 0.8
DEFAULT_X_KEY = "cls_tokens_ln"
DEFAULT_Y_KEY = "label"
DEFAULT_JSD_KEY = "jet_sdmass"

S1_BASIS = "/scope-vol/svd_results/QCD_massshift_cls_tokens_ln_svd.npz"
QCD_RES_BASIS = "/scope-vol/svd_results/QCD_res_cls_tokens_ln_svd.npz"

# Canonical sweep directories (relative to project root). Anything not listed
# here can still be processed via --sweep-dir.
DEFAULT_SWEEPS = [
    "runs/disco_pareto_pretrained",
    "runs/cure_disco_pareto__S1_massshift_ep40_k1",
    "runs/cure_disco_pareto__S1_massshift_ep40_k2",
    "runs/cure_disco_pareto__S1_massshift_ep40_k3",
    "runs/cure_disco_pareto__S1_massshift_k1",
    "runs/cure_disco_pareto__S1_massshift_k2",
    "runs/cure_disco_pareto__S1_massshift_k3",
]


def infer_svd_config(sweep_basename: str) -> tuple[Optional[str], Optional[int], Optional[float]]:
    """Map sweep dir basename → (svd_basis_path, svd_k, svd_variance).

    Returns (None, None, None) when no SVD projection is applied.
    Raises ValueError on unknown patterns.
    """
    name = sweep_basename
    if name.startswith("disco_pareto"):
        return None, None, None
    m = re.match(r"cure_disco_pareto__S1_massshift(?:_ep\d+)?_k(\d+)$", name)
    if m:
        return S1_BASIS, int(m.group(1)), None
    if name in ("cure_disco_pareto", "cure_forget_QCD_res"):
        return QCD_RES_BASIS, None, 0.95
    raise ValueError(f"Unknown sweep dir pattern: {name!r} — pass --svd-* explicitly")


def load_val_arrays(h5: str, *, seed: int = DEFAULT_SEED, train_frac: float = DEFAULT_TRAIN_FRAC):
    logger.info("Loading val features from %s", h5)
    x_all, y_all = load_h5_group(h5, "", DEFAULT_X_KEY, DEFAULT_Y_KEY, data_fraction=1.0)
    if y_all is None:
        raise RuntimeError("No labels in H5 — cannot reproduce stratified split")
    train_idx, val_idx = stratified_split(y_all, train_frac, seed)
    aux_all = load_aux_from_h5(h5, "", DEFAULT_JSD_KEY, len(x_all))
    x_va = x_all[val_idx]
    y_va = y_all[val_idx]
    aux_va = aux_all[val_idx]
    logger.info("val: x %s, y %s, jet_sdmass %s", x_va.shape, y_va.shape, aux_va.shape)
    return x_va, y_va, aux_va


def project_val(
    x_va: np.ndarray, basis_path: Optional[str], svd_k: Optional[int], svd_variance: Optional[float]
) -> np.ndarray:
    """Apply CURE-style SVD forget projection: x -> x - (x - mean) @ Vh.T @ Vh."""
    if basis_path is None:
        return x_va
    var = svd_variance if svd_variance is not None else 0.95
    Vh, mean, k_used = load_svd_basis(basis_path, svd_k, var)
    logger.info(
        "SVD forget: %s -> Vh shape %s, k_used=%d",
        os.path.basename(basis_path), tuple(Vh.shape), k_used,
    )
    Vh_t = torch.from_numpy(Vh.astype(np.float32))
    mean_t = torch.from_numpy(mean.astype(np.float32))
    x = torch.from_numpy(x_va.astype(np.float32))
    x_c = x - mean_t
    x_proj = x - (x_c @ Vh_t.T) @ Vh_t
    return x_proj.numpy()


def forward_probs(
    state_dict_path: str, x_va_np: np.ndarray, *, in_dim: int, num_classes: int, device: str,
    batch_size: int = 16384,
) -> np.ndarray:
    """Build PerpClassifier (input_is_post_ln=True, use_bias=True), load state_dict,
    return softmax probs over val set."""
    sd = torch.load(state_dict_path, map_location=device)
    if isinstance(sd, dict) and "model_state" in sd:
        sd = sd["model_state"]
    model = PerpClassifier(
        in_dim=in_dim, num_classes=num_classes,
        for_inference=False, use_bias=True, input_is_post_ln=True,
    )
    model.load_state_dict(sd)
    model.to(device).eval()

    out = np.empty((len(x_va_np), num_classes), dtype=np.float32)
    x_t = torch.from_numpy(x_va_np.astype(np.float32))
    with torch.no_grad():
        for i in range(0, len(x_t), batch_size):
            xb = x_t[i:i + batch_size].to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1) if num_classes > 1 else logits
            out[i:i + batch_size] = probs.cpu().numpy()
    return out


def reeval_run(
    run_dir: str, x_va_proj: np.ndarray, y_va: np.ndarray, aux_va: np.ndarray,
    *, device: str, dry_run: bool,
) -> Optional[dict]:
    best_pt = os.path.join(run_dir, "best.pt")
    bm_path = os.path.join(run_dir, "best_metrics.json")
    if not os.path.isfile(best_pt) or not os.path.isfile(bm_path):
        logger.warning("Skipping %s: missing best.pt or best_metrics.json", run_dir)
        return None
    with open(bm_path) as f:
        bm = json.load(f)
    in_dim = int(x_va_proj.shape[1])
    probs = forward_probs(best_pt, x_va_proj, in_dim=in_dim, num_classes=2, device=device)
    preds = probs.argmax(axis=1)
    # bin_width_gev=5.0 (the metric default) matches the DisCo paper's
    # 5 GeV bin width. n_bins is left unset so the function picks
    # round((mass_max - mass_min)/5) bins.
    new_met = compute_accuracy_and_jsd(
        preds, y_va, jsd_aux=aux_va, num_classes=2, probs=probs,
        qcd_label=0, hbb_label=1, target_misid_rate=0.01,
    )
    legacy_jsd = compute_jsd_argmax(preds, y_va, aux_va, qcd_label=0)

    # Sanity: eff@1pct should match the stored value closely (if not, val split is wrong).
    old_eff = bm.get("eff_at_1pct")
    new_eff = new_met.get("eff_at_1pct")
    if old_eff is not None and new_eff is not None:
        delta = abs(new_eff - old_eff)
        if delta > 1e-3:
            logger.warning(
                "[%s] eff@1pct mismatch: old=%.6f new=%.6f Δ=%.4g",
                run_dir, old_eff, new_eff, delta,
            )

    new_bm = dict(bm)
    # Preserve old jsd under a different key on first run only; subsequent re-runs
    # must not overwrite the genuine legacy value with an already-replaced field.
    if "jsd_argmax_legacy" not in new_bm and "jsd" in bm:
        new_bm["jsd_argmax_legacy"] = float(bm["jsd"])
    if "jsd" in new_met:
        new_bm["jsd"] = float(new_met["jsd"])
    else:
        new_bm.pop("jsd", None)
        logger.warning("[%s] new jsd is None (insufficient cut-passing QCD?); jsd key removed", run_dir)
    new_bm["jsd_argmax_recomputed"] = float(legacy_jsd)
    if new_eff is not None:
        new_bm["eff_at_1pct"] = float(new_eff)

    if dry_run:
        logger.info(
            "[%s] (dry-run) old jsd=%.5g -> new jsd=%.5g (argmax_recomp=%.5g)",
            run_dir, bm.get("jsd", float("nan")), new_bm.get("jsd", float("nan")), legacy_jsd,
        )
        return new_bm

    backup = bm_path + ".argmax_jsd_backup"
    if not os.path.exists(backup):
        with open(backup, "w") as f:
            json.dump(bm, f, indent=2)
    with open(bm_path, "w") as f:
        json.dump(new_bm, f, indent=2)
    logger.info(
        "[%s] jsd %.5g -> %.5g (legacy_recomputed=%.5g, eff@1pct=%.4f)",
        run_dir, bm.get("jsd", float("nan")), new_bm.get("jsd", float("nan")),
        legacy_jsd, new_bm.get("eff_at_1pct", float("nan")),
    )
    return new_bm


def collect_and_write_summary(sweep_dir: str, *, dry_run: bool) -> None:
    rows = []
    for name in sorted(os.listdir(sweep_dir)):
        if not name.startswith("lambda_"):
            continue
        run_dir = os.path.join(sweep_dir, name)
        bm_path = os.path.join(run_dir, "best_metrics.json")
        if not os.path.isfile(bm_path):
            continue
        try:
            with open(bm_path) as f:
                bm = json.load(f)
        except Exception:
            continue
        try:
            disco_lambda = float(name.replace("lambda_", ""))
        except ValueError:
            disco_lambda = None
        rows.append({"disco_lambda": disco_lambda, **bm})
    rows.sort(key=lambda r: (r.get("disco_lambda") is None, r.get("disco_lambda", 0.0)))
    if not rows:
        return
    keys = [
        "disco_lambda", "epoch", "acc", "loss", "eff_at_1pct",
        "jsd", "jsd_argmax_legacy", "jsd_argmax_recomputed",
        "disco_loss", "auc",
    ]
    keys = [k for k in keys if any(k in r for r in rows)]
    csv_path = os.path.join(sweep_dir, "summary.csv")
    if dry_run:
        logger.info("(dry-run) would write %s with %d rows, cols=%s", csv_path, len(rows), keys)
        return
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Wrote %s (%d rows)", csv_path, len(rows))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-dir", action="append", default=None,
                   help="Sweep directory (relative to project root). Repeatable. Default: canonical list.")
    p.add_argument("--h5", default=DEFAULT_H5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--svd-basis", default=None,
                   help="Override SVD basis path (used with --sweep-dir for unknown patterns)")
    p.add_argument("--svd-k", type=int, default=None,
                   help="Override SVD k (used with --sweep-dir for unknown patterns)")
    p.add_argument("--svd-variance", type=float, default=None,
                   help="Override SVD variance threshold (default 0.95)")
    args = p.parse_args()

    os.chdir(_PROJ)
    sweeps = args.sweep_dir or DEFAULT_SWEEPS
    sweeps = [s for s in sweeps if os.path.isdir(s)]
    if not sweeps:
        raise SystemExit("No sweep dirs found")

    x_va, y_va, aux_va = load_val_arrays(args.h5)

    # Cache projected val features per (basis, k, variance) tuple.
    proj_cache: dict[tuple, np.ndarray] = {}

    for sweep in sweeps:
        try:
            basis, svd_k, svd_var = infer_svd_config(os.path.basename(sweep.rstrip("/")))
        except ValueError as e:
            if args.svd_basis is None:
                logger.warning("%s: %s — skipping", sweep, e)
                continue
            basis = args.svd_basis
            svd_k = args.svd_k
            svd_var = args.svd_variance
        cache_key = (basis, svd_k, svd_var)
        if cache_key not in proj_cache:
            proj_cache[cache_key] = project_val(x_va, basis, svd_k, svd_var)
        x_va_proj = proj_cache[cache_key]

        logger.info("=== Re-evaluating sweep %s (basis=%s, k=%s) ===", sweep, basis, svd_k)
        any_done = False
        for name in sorted(os.listdir(sweep)):
            if not name.startswith("lambda_"):
                continue
            run_dir = os.path.join(sweep, name)
            if not os.path.isdir(run_dir):
                continue
            res = reeval_run(run_dir, x_va_proj, y_va, aux_va, device=args.device, dry_run=args.dry_run)
            any_done = any_done or (res is not None)
        if any_done:
            collect_and_write_summary(sweep, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
