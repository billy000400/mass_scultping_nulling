#!/usr/bin/env python3
"""Run disco loss scale (disco_lambda) sweep to map Pareto boundary on eff@1%bkg vs JSD.

Head init:
  Starts the linear head from the pretrained ParT head by default
  (--pretrained-head, via --ParT-model-path). This matches the cure_disco sweep's
  initialization so the comparison isolates the SVD-forget projection.
  Pass --no-pretrained-head to fall back to the previous random-init behaviour.

Usage:
  python scripts/run_disco_pareto_sweep.py [--lambdas 0 0.1 1 10 100] [--epochs 40]
  python scripts/run_disco_pareto_sweep.py --plot  # after sweep, generate scatter plot
  python scripts/run_disco_pareto_sweep.py --no-pretrained-head --save-dir runs/disco_pareto  # legacy behavior

Resume: By default, skips lambda values that already have best_metrics.json. Use --force to re-run.
Batch size: Default 4096 for faster training; use --batch-size 2048 if OOM.

Run from project root: cd /scope-vol/mass_perp_classifier

  # Match lambda grid from a prior sweep (e.g. after renaming runs/disco_pareto → *_legacy_*):
  python scripts/run_disco_pareto_sweep.py --legacy-summary runs/disco_pareto_legacy_20260422/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Same grid as legacy disco_pareto summary sweeps (see also --legacy-summary).
DEFAULT_LAMBDAS = [0, 0.1, 0.5, 1, 2, 5, 5.25, 5.5, 5.75, 6, 7, 8, 9, 10, 20, 50, 100, 200, 500, 1000]
# Default writes to runs/disco_pareto_pretrained (pretrained-head init), so the
# prior runs/disco_pareto (random-init) results are preserved side by side.
DEFAULT_SAVE_DIR = "runs/disco_pareto_pretrained"
DEFAULT_PART_MODEL = "/scope-vol/mass_perp_classifier/logs/no_bias/best.pt"
SUMMARY_CSV = "summary.csv"


def load_lambdas_from_summary_csv(path: str) -> list[float]:
    """Read ``disco_lambda`` column in row order (e.g. from a legacy ``summary.csv``)."""
    out: list[float] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out.append(float(row["disco_lambda"]))
    if not out:
        raise SystemExit(f"No rows in {path!r}")
    return out


def is_run_complete(save_dir: str, disco_lambda: float) -> bool:
    """True if this lambda already has best_metrics.json (completed run)."""
    run_dir = os.path.join(save_dir, f"lambda_{disco_lambda}")
    return os.path.isfile(os.path.join(run_dir, "best_metrics.json"))


def run_single(
    disco_lambda: float,
    save_dir: str,
    epochs: int,
    early_stop_patience: Optional[int],
    h5: str,
    device: str,
    batch_size: int,
    pretrained_head: bool,
    part_model_path: Optional[str],
    extra_args: list[str],
) -> bool:
    """Run one finetune experiment. Returns True on success."""
    run_dir = os.path.join(save_dir, f"lambda_{disco_lambda}")
    cmd = [
        sys.executable, "-m", "src.run_finetune",
        "--experiment", "disco",
        "--save-dir", run_dir,
        "--disco-lambda", str(disco_lambda),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--h5", h5,
        "--device", device,
    ]
    if pretrained_head:
        cmd.append("--pretrained-head")
        if part_model_path:
            cmd.extend(["--ParT-model-path", part_model_path])
    if early_stop_patience is not None:
        cmd.extend(["--early-stop-patience", str(early_stop_patience)])
    cmd.extend(extra_args)
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return result.returncode == 0


def collect_results(save_dir: str) -> list[dict]:
    """Collect best_metrics.json from each lambda run into a list of dicts."""
    results = []
    for name in sorted(os.listdir(save_dir)):
        if not name.startswith("lambda_"):
            continue
        run_dir = os.path.join(save_dir, name)
        if not os.path.isdir(run_dir):
            continue
        metrics_path = os.path.join(run_dir, "best_metrics.json")
        if not os.path.isfile(metrics_path):
            logger.warning("Missing %s", metrics_path)
            continue
        try:
            with open(metrics_path) as f:
                m = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", metrics_path, e)
            continue
        try:
            disco_lambda = float(name.replace("lambda_", ""))
        except ValueError:
            disco_lambda = None
        row = {"disco_lambda": disco_lambda, **m}
        results.append(row)
    return sorted(results, key=lambda r: (r.get("disco_lambda") is None, r.get("disco_lambda", 0)))


def write_summary_csv(results: list[dict], save_dir: str) -> str:
    """Write results to summary.csv. Returns path."""
    path = os.path.join(save_dir, SUMMARY_CSV)
    if not results:
        logger.warning("No results to write")
        return path
    keys = ["disco_lambda", "epoch", "acc", "loss", "eff_at_1pct", "jsd", "disco_loss"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in keys if any(k in r for r in results)], extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    logger.info("Wrote %s", path)
    return path


def plot_pareto(results: list[dict], save_dir: str, out_path: Optional[str]) -> None:
    """Generate scatter plot of eff@1%bkg vs JSD."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot")
        return
    eff = [r.get("eff_at_1pct") for r in results if r.get("eff_at_1pct") is not None]
    jsd = [r.get("jsd") for r in results if r.get("jsd") is not None]
    lambdas = [r.get("disco_lambda") for r in results]
    if not eff or not jsd or len(eff) != len(jsd):
        logger.warning("Insufficient data for plot")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(jsd, eff, c=lambdas, cmap="viridis", s=30)
    for i, lam in enumerate(lambdas):
        ax.annotate(f"{lam}", (jsd[i], eff[i]), fontsize=8, alpha=0.8)
    ax.set_xlabel("JSD (lower is better)")
    ax.set_ylabel("eff@1%bkg (higher is better)")
    ax.set_title("DisCo Pareto: eff@1%bkg vs JSD")
    plt.colorbar(scatter, ax=ax, label="disco_lambda")
    if out_path is None:
        out_path = os.path.join(save_dir, "pareto_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot to %s", out_path)


def main():
    p = argparse.ArgumentParser(description="DisCo lambda Pareto sweep")
    p.add_argument("--lambdas", nargs="+", type=float, default=DEFAULT_LAMBDAS,
                   help=f"Disco lambda values (default: built-in grid matching legacy Pareto). "
                   f"Ignored if --legacy-summary is set.")
    p.add_argument(
        "--legacy-summary",
        default=None,
        metavar="PATH",
        help="Path to a previous summary.csv; use its disco_lambda column in CSV order (overrides --lambdas).",
    )
    p.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--early-stop-patience", type=int, default=12)
    p.add_argument("--h5", default="/scope-vol/hidden_out/train_post_ln_HbbvsQCD_75_175_hiddens_w_jet_features.h5")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4096,
                   help="Batch size for training (default: 4096 for faster runs; lower if OOM)")
    head_group = p.add_mutually_exclusive_group()
    head_group.add_argument(
        "--pretrained-head",
        dest="pretrained_head",
        action="store_true",
        help="Initialize the linear head from --ParT-model-path (default). "
        "Gives an apples-to-apples init with cure_disco_pareto.",
    )
    head_group.add_argument(
        "--no-pretrained-head",
        dest="pretrained_head",
        action="store_false",
        help="Disable pretrained-head init (fall back to random init, the previous behavior).",
    )
    p.set_defaults(pretrained_head=True)
    p.add_argument(
        "--ParT-model-path",
        dest="part_model_path",
        default=DEFAULT_PART_MODEL,
        help=f"ParT checkpoint used for pretrained-head init (default: {DEFAULT_PART_MODEL})",
    )
    p.add_argument("--force", action="store_true",
                   help="Re-run even if best_metrics.json exists (default: skip completed runs)")
    p.add_argument("--plot", action="store_true", help="Generate Pareto scatter plot from existing results")
    p.add_argument("--plot-only", action="store_true", help="Only plot, do not run sweep")
    p.add_argument("extra", nargs="*", help="Extra args passed to run_finetune")
    args = p.parse_args()
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.legacy_summary:
        sp = args.legacy_summary
        if not os.path.isabs(sp):
            sp = os.path.join(proj_root, sp)
        if not os.path.isfile(sp):
            raise SystemExit(f"--legacy-summary not found: {sp}")
        args.lambdas = load_lambdas_from_summary_csv(sp)
        logger.info("Using %d lambdas from %s", len(args.lambdas), sp)

    os.chdir(proj_root)
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if args.plot_only:
        results = collect_results(save_dir)
        if results:
            write_summary_csv(results, save_dir)
            plot_pareto(results, save_dir, None)
        return

    if args.plot:
        results = collect_results(save_dir)
        if results:
            write_summary_csv(results, save_dir)
            plot_pareto(results, save_dir, None)
        return

    for lam in args.lambdas:
        if not args.force and is_run_complete(save_dir, lam):
            logger.info("Skipping lambda=%s (already complete)", lam)
            continue
        ok = run_single(
            disco_lambda=lam,
            save_dir=save_dir,
            epochs=args.epochs,
            early_stop_patience=args.early_stop_patience,
            h5=args.h5,
            device=args.device,
            batch_size=args.batch_size,
            pretrained_head=args.pretrained_head,
            part_model_path=args.part_model_path,
            extra_args=args.extra,
        )
        if not ok:
            logger.error("Run failed for lambda=%s", lam)

    results = collect_results(save_dir)
    write_summary_csv(results, save_dir)
    plot_pareto(results, save_dir, None)


if __name__ == "__main__":
    main()
