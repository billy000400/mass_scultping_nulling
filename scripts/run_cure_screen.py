#!/usr/bin/env python3
"""Fast inference-only CURE screen over (forget-basis, k) configs.

Loads model + latents ONCE, then for each (basis, k) applies the CURE forget
projection and evaluates eff@1% and JSD on the entire dataset (using the same
metrics as the finetune sweeps). Writes a summary CSV and scatter plot.

Usage (from project root):
  python scripts/run_cure_screen.py [--configs ...]
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Make "src" importable when run from the project root.
HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.core.data import load_h5_group, load_aux_from_h5, make_loader  # noqa: E402
from src.core.metrics import compute_accuracy_and_jsd  # noqa: E402
from src.core.model import PerpClassifier, load_pretrained_part_weights  # noqa: E402
from src.core.projections import load_svd_basis  # noqa: E402
from src.core.utils import infer_num_classes, set_seed  # noqa: E402
from src.experiments.no_finetune import run_inference  # noqa: E402

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_H5 = "/scope-vol/hidden_out/train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5"
DEFAULT_PART = "/scope-vol/mass_perp_classifier/logs/no_bias/best.pt"

DEFAULT_CONFIGS: List[Tuple[str, str, List[int]]] = [
    # (nickname, basis-path, k-list)
    ("S1_massshift", "/scope-vol/svd_results/QCD_massshift_cls_tokens_ln_svd.npz",
     [1, 2, 3, 5, 7, 10]),
    ("S2_massshift_minusHbb", "/scope-vol/svd_results/QCD_massshift_minusHbb_cls_tokens_ln_svd.npz",
     [3, 7, 15]),
    ("S3_ols_deflated", "/scope-vol/svd_results/QCD_olsmass_cls_tokens_ln_svd.npz",
     [3, 7, 15]),
    ("S4_full_qcd_svd", "/scope-vol/svd_results/QCD_res_cls_tokens_ln_svd.npz",
     [1, 3, 5, 7, 10]),
    # k=0 no-forget baseline, same code path (uses alpha=0 equivalently).
]


def parse_cfg(s: str) -> Tuple[str, str, List[int]]:
    """Parse a --config arg of the form 'label:basis.npz:k1,k2,k3'."""
    parts = s.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 'label:path:k1,k2...', got {s!r}")
    label, path, ks = parts
    return label.strip(), path.strip(), [int(k) for k in ks.split(",") if k.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", default=DEFAULT_H5)
    ap.add_argument("--part-model", default=DEFAULT_PART, dest="part_model")
    ap.add_argument("--config", action="append", type=parse_cfg, default=None,
                    help="Repeatable 'label:path:k1,k2,...' (default: DEFAULT_CONFIGS)")
    ap.add_argument("--save-dir", default="runs/cure_screen")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--include-noforget", action="store_true", default=True,
                    help="Also evaluate the pretrained head with no projection (k=0)")
    args = ap.parse_args()

    cfgs = args.config if args.config else DEFAULT_CONFIGS
    save_dir = (args.save_dir if os.path.isabs(args.save_dir)
                else os.path.join(str(PROJ), args.save_dir))
    os.makedirs(save_dir, exist_ok=True)

    set_seed(1337)
    logger.info("Loading latent from %s", args.h5)
    x, y = load_h5_group(args.h5, "", "cls_tokens_ln", "label", data_fraction=1.0)
    n = len(x)
    jsd_aux = load_aux_from_h5(args.h5, "", "jet_sdmass", n)
    in_dim = int(x.shape[1])

    num_classes = infer_num_classes(y)
    logger.info("n=%d in_dim=%d num_classes=%d", n, in_dim, num_classes)

    loader = make_loader(x, None, batch_size=args.batch_size, shuffle=False)

    state = torch.load(args.part_model, map_location="cpu")
    sd = state.get("model_state", state)
    fc_w = None
    for k, v in sd.items():
        if "fc" in k and k.endswith(".weight") and v.ndim == 2:
            fc_w = v
            break
    if fc_w is None:
        raise RuntimeError("no FC weights found in " + args.part_model)
    out_dim, ckpt_in = fc_w.shape
    if ckpt_in != in_dim:
        logger.warning("Input dim mismatch ckpt=%d data=%d", ckpt_in, in_dim)
    pretrained = load_pretrained_part_weights(
        args.part_model, in_dim=ckpt_in, num_classes=out_dim, device=args.device,
    )
    model = PerpClassifier(
        in_dim=ckpt_in, num_classes=out_dim,
        for_inference=True, use_pretrained_norm_fc=True,
        pretrained_state_dict=pretrained, input_is_post_ln=True,
    ).to(args.device)

    results: list[dict] = []

    if args.include_noforget:
        logger.info("eval: no-forget baseline")
        scores = run_inference(model, loader, args.device)
        preds = np.argmax(scores, axis=1)
        met = compute_accuracy_and_jsd(preds, y, jsd_aux=jsd_aux,
                                       num_classes=out_dim, probs=scores)
        row = {"strategy": "no_forget", "k": 0, **met}
        logger.info("  -> eff@1%%=%.4f jsd=%.5f acc=%.4f",
                    met.get("eff_at_1pct", float("nan")),
                    met.get("jsd", float("nan")), met.get("acc", float("nan")))
        results.append(row)

    for label, path, ks in cfgs:
        if not os.path.isfile(path):
            logger.warning("missing %s, skipping %s", path, label)
            continue
        for k in ks:
            Vh_np, mean_np, k_eff = load_svd_basis(path, k=k)
            forget_basis = torch.from_numpy(Vh_np).to(args.device)
            forget_mean = torch.from_numpy(mean_np).to(args.device)
            logger.info("eval: %s k=%d (basis rows=%d)", label, k, k_eff)
            scores = run_inference(
                model, loader, args.device,
                svd_forget_basis=forget_basis, svd_forget_mean=forget_mean,
            )
            preds = np.argmax(scores, axis=1)
            met = compute_accuracy_and_jsd(
                preds, y, jsd_aux=jsd_aux, num_classes=out_dim, probs=scores,
            )
            row = {"strategy": label, "k": k_eff, **met}
            logger.info("  -> eff@1%%=%.4f jsd=%.5f acc=%.4f",
                        met.get("eff_at_1pct", float("nan")),
                        met.get("jsd", float("nan")), met.get("acc", float("nan")))
            results.append(row)

    csv_path = os.path.join(save_dir, "summary.csv")
    keys = ["strategy", "k", "acc", "balanced_acc", "jsd", "eff_at_1pct"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    logger.info("wrote %s", csv_path)

    # Scatter
    fig, ax = plt.subplots(figsize=(9, 6))
    style = {}
    for r in results:
        s = r["strategy"]
        if s not in style:
            style[s] = len(style)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(10, len(style))))
    for s, i in style.items():
        sub = [r for r in results if r["strategy"] == s]
        ax.plot([r["jsd"] for r in sub], [r["eff_at_1pct"] for r in sub],
                "o-", color=cmap[i % 10], label=s, markersize=8)
        for r in sub:
            ax.annotate(f"k={r['k']}", (r["jsd"], r["eff_at_1pct"]),
                        fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("JSD (lower is better)")
    ax.set_ylabel("eff@1% (higher is better)")
    ax.set_title("CURE inference-only screen (pretrained head, no finetune)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    out_png = os.path.join(save_dir, "screen_scatter.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_png)


if __name__ == "__main__":
    main()
