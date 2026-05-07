#!/usr/bin/env python3
"""Outer driver: sweep S1 mass-shift basis at multiple k values, each with a
DisCo-lambda Pareto sub-sweep, all sharing the same canonical training protocol.

For each k in --ks, calls scripts/run_cure_disco_pareto_sweep.py with:
    --save-dir runs/cure_disco_pareto__<basis_tag>_k<k>
    --svd-forget <basis>
    --lambdas <grid>
    --epochs <epochs>
    extra args: --svd-k <k>

Resumes safely (the sub-sweep skips lambdas with existing best_metrics.json).
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent

DEFAULT_BASIS = "/scope-vol/svd_results/QCD_massshift_cls_tokens_ln_svd.npz"
DEFAULT_LAMBDAS = [0.0, 0.5, 1.0, 2.0, 5.0, 7.0]
DEFAULT_KS = [1, 2, 3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basis", default=DEFAULT_BASIS,
                    help=f"SVD basis .npz (default: {DEFAULT_BASIS})")
    ap.add_argument("--basis-tag", default="S1_massshift",
                    help="Short tag for save-dir naming")
    ap.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS)
    ap.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--early-stop-patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-dir-root", default="runs",
                    help="Parent dir for save-dirs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sweep_script = PROJ / "scripts" / "run_cure_disco_pareto_sweep.py"
    save_dir_root = (args.save_dir_root if os.path.isabs(args.save_dir_root)
                     else str(PROJ / args.save_dir_root))

    for k in args.ks:
        save_dir = os.path.join(save_dir_root, f"cure_disco_pareto__{args.basis_tag}_k{k}")
        cmd = [
            sys.executable, str(sweep_script),
            "--save-dir", save_dir,
            "--svd-forget", args.basis,
            "--lambdas", *(str(l) for l in args.lambdas),
            "--epochs", str(args.epochs),
            "--early-stop-patience", str(args.early_stop_patience),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--",  # separator: everything after goes to extra (forwarded to run_finetune)
            "--svd-k", str(k),
        ]
        logger.info("k=%d  save_dir=%s", k, save_dir)
        logger.info("  cmd: %s", " ".join(cmd))
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=str(PROJ))
        if result.returncode != 0:
            logger.error("sweep failed for k=%d (rc=%d)", k, result.returncode)


if __name__ == "__main__":
    main()
