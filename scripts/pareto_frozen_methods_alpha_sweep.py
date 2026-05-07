#!/usr/bin/env python3
"""Frozen-head Pareto: alpha-sweep at k=1 for three direction sources.

Compares, with the pretrained linear head frozen, how each of the following
single directions traces an Eff@1% vs JSD frontier as the projection
strength alpha is dialled from 0 to 1:

    - Mass-binned PCA (S1)  -- /scope-vol/svd_results/QCD_massshift_cls_tokens_ln_svd.npz
    - Plain PCA (S4)        -- /scope-vol/svd_results/QCD_res_cls_tokens_ln_svd.npz
    - RAV (S3, unit OLS)    -- /scope-vol/svd_results/QCD_olsmass_cls_tokens_ln_svd.npz

Outputs CSV + plot with three curves anchored at the unconstrained-ParT
point (alpha = 0).

Usage (from project root):
    python scripts/pareto_frozen_methods_alpha_sweep.py
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.core.data import load_h5_group, load_aux_from_h5, make_loader  # noqa: E402
from src.core.metrics import compute_accuracy_and_jsd  # noqa: E402
from src.core.model import PerpClassifier, load_pretrained_part_weights  # noqa: E402
from src.core.utils import infer_num_classes, set_seed  # noqa: E402
from src.experiments.no_finetune import run_inference  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_METHODS: List[Tuple[str, str, str]] = [
    # (label, basis-path, plot-color)
    ("S1 (mass-binned PCA)",
     "/scope-vol/svd_results/QCD_massshift_cls_tokens_ln_svd.npz",
     "#1f77b4"),
    ("plain PCA",
     "/scope-vol/svd_results/QCD_res_cls_tokens_ln_svd.npz",
     "#2ca02c"),
    ("RAV (unit OLS)",
     "/scope-vol/svd_results/QCD_olsmass_cls_tokens_ln_svd.npz",
     "#ff7f0e"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--h5", default="/scope-vol/hidden_out/train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5",
    )
    ap.add_argument(
        "--part-model", default="/scope-vol/mass-sculpting-nulling/logs/no_bias/best.pt",
        dest="part_model",
    )
    ap.add_argument("--alpha-grid", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--k", type=int, default=1,
                    help="number of directions to project out per method (default 1)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/scope-vol/mass-sculpting-nulling/results"))
    ap.add_argument("--stem", default="pareto_frozen_methods_alpha_sweep")
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

    loader = make_loader(x_full, None, batch_size=args.batch_size, shuffle=False)

    # Single anchor: alpha=0 (unconstrained ParT)
    logger.info("baseline (alpha=0, no projection)")
    scores = run_inference(model, loader, args.device)
    preds = np.argmax(scores, axis=1)
    met = compute_accuracy_and_jsd(preds, y_full, jsd_aux=jsd_aux,
                                   num_classes=out_dim, probs=scores)
    eff0 = float(met.get("eff_at_1pct", float("nan")))
    jsd0 = float(met.get("jsd", float("nan")))
    logger.info("  baseline  Eff@1%%=%.4f  JSD=%.4f", eff0, jsd0)

    rows: list[dict] = []
    rows.append({"method": "unconstrained_ParT", "alpha": 0.0,
                 "eff_at_1pct": eff0, "jsd": jsd0})

    for label, path, _ in DEFAULT_METHODS:
        d = np.load(path)
        Vh_full = np.asarray(d["Vh"], dtype=np.float32)
        mean_full = np.asarray(d["mean"], dtype=np.float32)
        kk = min(args.k, Vh_full.shape[0])
        Vh_t = torch.from_numpy(Vh_full[:kk]).to(args.device)
        mean_t = torch.from_numpy(mean_full).to(args.device)
        for alpha in args.alpha_grid:
            if alpha == 0.0:
                rows.append({"method": label, "alpha": 0.0,
                             "eff_at_1pct": eff0, "jsd": jsd0})
                logger.info("  %s alpha=%.2f -> baseline", label, alpha)
                continue
            scores = run_inference(
                model, loader, args.device,
                svd_forget_basis=Vh_t, svd_forget_mean=mean_t, alpha=alpha,
            )
            preds = np.argmax(scores, axis=1)
            met = compute_accuracy_and_jsd(
                preds, y_full, jsd_aux=jsd_aux, num_classes=out_dim, probs=scores,
            )
            eff = float(met.get("eff_at_1pct", float("nan")))
            jsd = float(met.get("jsd", float("nan")))
            rows.append({"method": label, "alpha": alpha,
                         "eff_at_1pct": eff, "jsd": jsd})
            logger.info("  %s alpha=%.2f  Eff@1%%=%.4f  JSD=%.4f",
                        label, alpha, eff, jsd)

    # CSV
    csv_path = args.out_dir / f"{args.stem}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "alpha", "eff_at_1pct", "jsd"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info("wrote %s", csv_path)

    # Plot
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for label, path, color in DEFAULT_METHODS:
        sub = sorted([r for r in rows if r["method"] == label],
                     key=lambda r: r["alpha"])
        jsds = np.array([r["jsd"] for r in sub])
        effs = np.array([r["eff_at_1pct"] for r in sub])
        # Detect collapse (all alpha values give the same (jsd, eff)).
        collapsed = (np.ptp(jsds) < 1e-5) and (np.ptp(effs) < 1e-5)
        if collapsed:
            ax.scatter([jsds[0]], [effs[0]], marker="x", s=80, color=color,
                       zorder=4, label=f"{label} (k={args.k}) — α-sweep collapses to baseline")
        else:
            ax.plot(jsds, effs, "o-", color=color, lw=1.4, ms=5,
                    label=f"{label} (k={args.k})")
            for r in sub:
                if r["alpha"] in (0.0, 0.5, 1.0):
                    ax.annotate(f"α={r['alpha']:.1f}",
                                (r["jsd"], r["eff_at_1pct"]),
                                fontsize=7, alpha=0.6,
                                xytext=(3, 3), textcoords="offset points")
    ax.scatter([jsd0], [eff0], marker="*", s=160, color="black",
               zorder=5, label="unconstrained ParT (α=0)")
    ax.set_xlabel("JSD (mass sculpting; lower = better)")
    ax.set_ylabel("Eff@1% (Hbb tag eff at 1% QCD mistag; higher = better)")
    ax.set_title(f"Frozen-head α-sweep, k={args.k}: direction-source comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    png = args.out_dir / f"{args.stem}.png"
    pdf = args.out_dir / f"{args.stem}.pdf"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s and .pdf", png)


if __name__ == "__main__":
    main()
