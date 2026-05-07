#!/usr/bin/env python3
"""Compute AUC for every best.pt in the Phase-4 ep40 runs + DisCo-only pretrained.

For each lambda subdir that has a best.pt:
  - Loads the val split (same seed/split as training).
  - Applies SVD projection for cure_disco runs.
  - Runs inference, computes AUC.
  - Updates best_metrics.json with {"auc": <value>}.

After processing all subdirs, rewrites summary.csv for each sweep root
(adding an "auc" column) and prints a table.

Usage:
  python scripts/compute_auc_all_runs.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.experiments.finetune import _load_svd
from src.core.model import PerpClassifier
from src.core.projections import project_svd
from src.core.data import load_h5_group, stratified_split

H5 = "/scope-vol/hidden_out/train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5"
SVD_S1 = "/scope-vol/svd_results/QCD_massshift_cls_tokens_ln_svd.npz"
SEED = 1337
TRAIN_FRAC = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 4096

SWEEP_ROOTS = [
    # (sweep_dir, svd_k_or_None)
    ("runs/cure_disco_pareto__S1_massshift_ep40_k1", 1),
    ("runs/cure_disco_pareto__S1_massshift_ep40_k2", 2),
    ("runs/cure_disco_pareto__S1_massshift_ep40_k3", 3),
    ("runs/disco_pareto_pretrained", None),
]

SUMMARY_KEYS = ["disco_lambda", "epoch", "acc", "loss", "eff_at_1pct", "jsd", "auc", "disco_loss"]


def compute_auc(scores: np.ndarray, targets: np.ndarray, pos_label: int = 1) -> float:
    mask = (targets == 0) | (targets == pos_label)
    y = (targets[mask] == pos_label).astype(np.float64)
    s = scores[mask]
    order = np.argsort(s)[::-1]
    y_s = y[order]
    n_pos = y_s.sum()
    n_neg = len(y_s) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = np.cumsum(y_s)
    fp = np.cumsum(1.0 - y_s)
    tpr = np.concatenate([[0.0], tp / n_pos])
    fpr = np.concatenate([[0.0], fp / n_neg])
    return float(np.trapz(tpr, fpr))


def load_val_data() -> tuple[np.ndarray, np.ndarray]:
    print("Loading val data …")
    x_all, y_all = load_h5_group(H5, "", "cls_tokens_ln", "label")
    _, val_idx = stratified_split(y_all, TRAIN_FRAC, SEED)
    x_va = x_all[val_idx]
    y_va = y_all[val_idx]
    perm = np.random.RandomState(SEED + 1).permutation(len(x_va))
    return x_va[perm], y_va[perm]


def infer(model: torch.nn.Module, x: np.ndarray,
          svd_f, svd_fm, svd_r, svd_rm) -> np.ndarray:
    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(x), BATCH):
            xb = torch.from_numpy(x[start:start + BATCH]).to(DEVICE)
            if svd_f is not None:
                xb = project_svd(xb, svd_f, svd_fm, svd_r, svd_rm)
            out = model(xb)
            if isinstance(out, (tuple, list)):
                out = out[0]
            probs = torch.softmax(out, dim=-1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def process_sweep(sweep_dir: str, svd_k: int | None,
                  x_va: np.ndarray, y_va: np.ndarray) -> list[dict]:
    abs_dir = os.path.join(ROOT, sweep_dir)
    if not os.path.isdir(abs_dir):
        print(f"  SKIP (missing): {sweep_dir}")
        return []

    # Load SVD projection once per sweep
    svd_f = svd_fm = svd_r = svd_rm = None
    if svd_k is not None:
        svd_f, svd_fm, svd_r, svd_rm = _load_svd(SVD_S1, None, svd_k, 0.95, DEVICE)

    in_dim = x_va.shape[1]
    num_classes = len(np.unique(y_va))

    results = []
    lambda_dirs = sorted(
        [d for d in os.listdir(abs_dir) if d.startswith("lambda_")],
        key=lambda s: float(s.split("_", 1)[1]),
    )

    for ldir in lambda_dirs:
        run_dir = os.path.join(abs_dir, ldir)
        best_pt = os.path.join(run_dir, "best.pt")
        metrics_path = os.path.join(run_dir, "best_metrics.json")
        if not os.path.isfile(best_pt) or not os.path.isfile(metrics_path):
            print(f"  SKIP (no best.pt or metrics): {ldir}")
            continue

        with open(metrics_path) as f:
            m = json.load(f)

        lam = float(ldir.split("_", 1)[1])

        # Load model
        model = PerpClassifier(
            in_dim=in_dim,
            num_classes=num_classes,
            for_inference=False,
            use_pretrained_norm_fc=False,
            input_is_post_ln=True,
        ).to(DEVICE)
        state = torch.load(best_pt, map_location=DEVICE)
        model.load_state_dict(state)

        probs = infer(model, x_va, svd_f, svd_fm, svd_r, svd_rm)
        auc = compute_auc(probs[:, 1], y_va)

        m["auc"] = round(auc, 6)
        with open(metrics_path, "w") as f:
            json.dump(m, f, indent=2)

        row = {"disco_lambda": lam, **m}
        results.append(row)
        print(f"  {ldir}: auc={auc:.4f}  eff={m.get('eff_at_1pct', 'n/a'):.4f}  jsd={m.get('jsd', 'n/a'):.5f}")

    # Rewrite summary.csv
    if results:
        csv_path = os.path.join(abs_dir, "summary.csv")
        keys = [k for k in SUMMARY_KEYS if any(k in r for r in results)]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        print(f"  -> {csv_path}")

    return results


def main() -> None:
    x_va, y_va = load_val_data()
    print(f"Val set: {len(x_va)} samples, in_dim={x_va.shape[1]}, device={DEVICE}\n")

    for sweep_dir, svd_k in SWEEP_ROOTS:
        tag = sweep_dir.split("/")[-1]
        print(f"=== {tag} (svd_k={svd_k}) ===")
        process_sweep(sweep_dir, svd_k, x_va, y_va)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
