#!/usr/bin/env python3
"""k-ablation table for S1 (mass-binned PCA) at alpha=1, no finetune.

Justifies the choice of k for the main results. For each k in --k-grid:
  1. Apply CURE-style nulling of S1's top-k directions to the QCD+Hbb post-LN
     CLS tokens (alpha=1).
  2. Compute the linear-probe R^2 of jet_sdmass from h' on a held-out QCD
     split (50/50 train/test, ridge=1e-3).
  3. Run the pretrained linear head on h' across the full dataset, compute
     Eff@1% (Hbb efficiency at 1% QCD mistag) and JSD (mean per-class JSD
     of true-QCD mass histograms vs inclusive QCD reference).

One data load, one model load. Outputs:
  - CSV table (one row per k)
  - markdown table dumped to stdout (paste-ready for the paper)
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.core.data import load_h5_group, load_aux_from_h5, make_loader  # noqa: E402
from src.core.metrics import compute_accuracy_and_jsd  # noqa: E402
from src.core.model import PerpClassifier, load_pretrained_part_weights  # noqa: E402
from src.core.projections import load_svd_basis, project_svd  # noqa: E402
from src.core.utils import infer_num_classes, set_seed  # noqa: E402
from src.experiments.no_finetune import run_inference  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

QCD_LABEL = 0


def _ridge_r2_train_test(
    X_tr, m_tr, X_te, m_te, ridge: float = 1e-3,
) -> float:
    mu = X_tr.mean(0, keepdims=True)
    Xc_tr = X_tr - mu
    Xc_te = X_te - mu
    mu_m = m_tr.mean()
    s_m = m_tr.std() + 1e-12
    mc_tr = (m_tr - mu_m) / s_m
    A = Xc_tr.T @ Xc_tr + ridge * np.eye(Xc_tr.shape[1])
    w = np.linalg.solve(A, Xc_tr.T @ mc_tr)
    pred = Xc_te @ w
    targ = (m_te - mu_m) / s_m
    ss_r = float(np.sum((targ - pred) ** 2))
    ss_t = float(np.sum((targ - targ.mean()) ** 2))
    return 1.0 - ss_r / ss_t


def _null_topk(X: np.ndarray, Vh: np.ndarray, mean: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or Vh.shape[0] == 0:
        return X
    V = Vh[:k]
    Xc = X - mean
    return X - (Xc @ V.T) @ V


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--h5", default="/scope-vol/hidden_out/train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5",
    )
    ap.add_argument(
        "--basis", default="/scope-vol/svd_results/QCD_massshift_cls_tokens_ln_svd.npz",
    )
    ap.add_argument(
        "--part-model", default="/scope-vol/mass-sculpting-nulling/logs/no_bias/best.pt",
        dest="part_model",
    )
    ap.add_argument("--k-grid", type=int, nargs="+",
                    default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--probe-max-qcd", type=int, default=400_000)
    ap.add_argument("--probe-seed", type=int, default=42)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/scope-vol/mass-sculpting-nulling/results"))
    ap.add_argument("--stem", default="k_ablation_S1_no_finetune")
    args = ap.parse_args()

    set_seed(1337)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Load latents from %s", args.h5)
    x_full, y_full = load_h5_group(args.h5, "", "cls_tokens_ln", "label",
                                   data_fraction=1.0)
    n = len(x_full)
    jsd_aux = load_aux_from_h5(args.h5, "", "jet_sdmass", n)
    in_dim = int(x_full.shape[1])
    num_classes = infer_num_classes(y_full)
    logger.info("n=%d in_dim=%d num_classes=%d", n, in_dim, num_classes)

    # Pretrained linear head
    state = torch.load(args.part_model, map_location="cpu")
    sd = state.get("model_state", state)
    fc_w = next((v for k, v in sd.items()
                 if "fc" in k and k.endswith(".weight") and v.ndim == 2), None)
    if fc_w is None:
        raise RuntimeError("no FC weights found in " + args.part_model)
    out_dim = fc_w.shape[0]
    pretrained = load_pretrained_part_weights(
        args.part_model, in_dim=in_dim, num_classes=out_dim, device=args.device,
    )
    model = PerpClassifier(
        in_dim=in_dim, num_classes=out_dim,
        for_inference=True, use_pretrained_norm_fc=True,
        pretrained_state_dict=pretrained, input_is_post_ln=True,
    ).to(args.device)

    # Basis
    Vh_full = np.load(args.basis)["Vh"].astype(np.float64)
    mean_full = np.load(args.basis)["mean"].astype(np.float64)

    # Held-out QCD split for the linear probe
    x_np = np.asarray(x_full, dtype=np.float64) if not isinstance(x_full, np.ndarray) \
        else x_full.astype(np.float64)
    y_np = np.asarray(y_full).reshape(-1).astype(np.int64)
    m_np = np.asarray(jsd_aux).reshape(-1).astype(np.float64)
    qcd_idx = np.where(y_np == QCD_LABEL)[0]
    if qcd_idx.size > args.probe_max_qcd:
        qcd_idx = qcd_idx[:args.probe_max_qcd]
    rng = np.random.default_rng(args.probe_seed)
    perm = rng.permutation(qcd_idx.size)
    half = qcd_idx.size // 2
    qcd_tr = qcd_idx[perm[:half]]
    qcd_te = qcd_idx[perm[half:]]
    logger.info("probe split: train=%d test=%d (QCD only)", len(qcd_tr), len(qcd_te))

    # Loader for the classifier head (full data, no shuffle).
    loader = make_loader(x_full, None, batch_size=args.batch_size, shuffle=False)

    rows = []
    for k in args.k_grid:
        # ---- linear probe R^2 on held-out QCD ----
        X_tr = _null_topk(x_np[qcd_tr], Vh_full, mean_full, k)
        X_te = _null_topk(x_np[qcd_te], Vh_full, mean_full, k)
        r2 = _ridge_r2_train_test(X_tr, m_np[qcd_tr],
                                  X_te, m_np[qcd_te], args.ridge)

        # ---- pretrained head: Eff@1% and JSD ----
        if k == 0:
            scores = run_inference(model, loader, args.device)
        else:
            Vh_k = torch.from_numpy(Vh_full[:k].astype(np.float32)).to(args.device)
            mean_t = torch.from_numpy(mean_full.astype(np.float32)).to(args.device)
            scores = run_inference(
                model, loader, args.device,
                svd_forget_basis=Vh_k, svd_forget_mean=mean_t,
            )
        preds = np.argmax(scores, axis=1)
        met = compute_accuracy_and_jsd(
            preds, y_full, jsd_aux=jsd_aux, num_classes=out_dim, probs=scores,
        )
        eff = float(met.get("eff_at_1pct", float("nan")))
        jsd = float(met.get("jsd", float("nan")))

        rows.append({"k": k, "r2_test": r2, "eff_at_1pct": eff, "jsd": jsd})
        logger.info("k=%d  R^2=%+.4f  Eff@1%%=%.4f  JSD=%.4f", k, r2, eff, jsd)

    # CSV
    csv_path = args.out_dir / f"{args.stem}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "r2_test", "eff_at_1pct", "jsd"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info("wrote %s", csv_path)

    # Markdown
    md = ["| k | linear $R^2(m\\,|\\,h')$ | Eff@1% | JSD |",
          "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['k']} | {r['r2_test']:+.4f} | {r['eff_at_1pct']:.4f} | {r['jsd']:.4f} |")
    md_str = "\n".join(md)
    md_path = args.out_dir / f"{args.stem}.md"
    md_path.write_text(md_str + "\n")
    logger.info("wrote %s", md_path)
    print()
    print(md_str)


if __name__ == "__main__":
    main()
