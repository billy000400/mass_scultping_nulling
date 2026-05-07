#!/usr/bin/env python3
"""Frozen-head Pareto: sweep S1 projection strength alpha at fixed k.

For each (k, alpha) in (--k-grid, --alpha-grid):
  - Apply CURE-style nulling of the top-k S1 directions with strength alpha.
  - Run the pretrained linear head (no finetune).
  - Record Eff@1% and JSD.

Outputs CSV + Eff@1% vs JSD plot (one curve per k, parameterised by alpha).

Baselines on the plot:
  - alpha = 0  is the unconstrained-ParT point (no projection); appears as
    the leftmost / largest-Eff anchor of every k-curve and is also drawn
    once explicitly.

Usage (from project root):
  python scripts/pareto_frozen_alpha_sweep.py
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

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
    ap.add_argument("--k-grid", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--alpha-grid", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path("/scope-vol/mass-sculpting-nulling/results"),
    )
    ap.add_argument("--stem", default="pareto_frozen_alpha_sweep")
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

    Vh_full = np.load(args.basis)["Vh"].astype(np.float32)
    mean_full = np.load(args.basis)["mean"].astype(np.float32)
    mean_t = torch.from_numpy(mean_full).to(args.device)

    loader = make_loader(x_full, None, batch_size=args.batch_size, shuffle=False)

    rows = []

    # Single anchor point: alpha=0, no projection (any k is equivalent).
    logger.info("baseline (alpha=0, no projection)")
    scores = run_inference(model, loader, args.device)
    preds = np.argmax(scores, axis=1)
    met = compute_accuracy_and_jsd(preds, y_full, jsd_aux=jsd_aux,
                                   num_classes=out_dim, probs=scores)
    eff0 = float(met.get("eff_at_1pct", float("nan")))
    jsd0 = float(met.get("jsd", float("nan")))
    rows.append({"k": 0, "alpha": 0.0, "eff_at_1pct": eff0, "jsd": jsd0})
    logger.info("  baseline  Eff@1%%=%.4f  JSD=%.4f", eff0, jsd0)

    for k in args.k_grid:
        Vh_k = torch.from_numpy(Vh_full[:k]).to(args.device)
        for alpha in args.alpha_grid:
            if alpha == 0.0:
                # Same as the baseline; record the (k, 0) point so each
                # k-curve has an explicit alpha=0 anchor.
                rows.append({"k": k, "alpha": 0.0,
                             "eff_at_1pct": eff0, "jsd": jsd0})
                logger.info("  k=%d alpha=%.2f  -> baseline", k, alpha)
                continue
            scores = run_inference(
                model, loader, args.device,
                svd_forget_basis=Vh_k, svd_forget_mean=mean_t, alpha=alpha,
            )
            preds = np.argmax(scores, axis=1)
            met = compute_accuracy_and_jsd(
                preds, y_full, jsd_aux=jsd_aux, num_classes=out_dim, probs=scores,
            )
            eff = float(met.get("eff_at_1pct", float("nan")))
            jsd = float(met.get("jsd", float("nan")))
            rows.append({"k": k, "alpha": alpha,
                         "eff_at_1pct": eff, "jsd": jsd})
            logger.info("  k=%d alpha=%.2f  Eff@1%%=%.4f  JSD=%.4f",
                        k, alpha, eff, jsd)

    # CSV
    csv_path = args.out_dir / f"{args.stem}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "alpha", "eff_at_1pct", "jsd"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    logger.info("wrote %s", csv_path)

    # Plot
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    cmap = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#9467bd", 5: "#8c564b"}
    for k in args.k_grid:
        sub = sorted(
            [r for r in rows if r["k"] == k],
            key=lambda r: r["alpha"],
        )
        jsds = [r["jsd"] for r in sub]
        effs = [r["eff_at_1pct"] for r in sub]
        ax.plot(jsds, effs, "o-", color=cmap.get(k, "gray"),
                lw=1.4, ms=5, label=f"S1 frozen, k={k}")
        # alpha annotations on every other point
        for i, r in enumerate(sub):
            if i % 2 == 0:
                ax.annotate(f"α={r['alpha']:.1f}",
                            (r["jsd"], r["eff_at_1pct"]),
                            fontsize=7, alpha=0.6,
                            xytext=(3, 3), textcoords="offset points")
    ax.scatter([jsd0], [eff0], marker="*", s=140, color="black",
               zorder=5, label=f"unconstrained ParT (α=0)")
    ax.set_xlabel("JSD (mass sculpting; lower = better)")
    ax.set_ylabel("Eff@1% (Hbb tag eff at 1% QCD mistag; higher = better)")
    # No in-figure title; the LaTeX caption is the title (ICML rule).
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    png = args.out_dir / f"{args.stem}.png"
    pdf = args.out_dir / f"{args.stem}.pdf"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s and .pdf", png)


if __name__ == "__main__":
    main()
