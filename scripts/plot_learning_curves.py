#!/usr/bin/env python3
"""Plot learning curves (loss, acc, eff@1%bkg, JSD) vs epoch for a Pareto sweep.

Reads TensorBoard scalar logs under ``<run_dir>/lambda_*/tb``.

Modes:
  * Aggregated (default): one figure overlaying all lambdas, coloured on a log
    scale, saved to ``<run_dir>/learning_curves.png``.
  * Per-lambda (``--per-lambda``): one figure per lambda saved next to its tb
    directory at ``<run_dir>/lambda_<X>/learning_curves.png``.

Usage:
  python scripts/plot_learning_curves.py runs/disco_pareto
  python scripts/plot_learning_curves.py runs/disco_pareto --per-lambda
  python scripts/plot_learning_curves.py runs/disco_pareto --split  # train and val side-by-side
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

METRICS = [
    ("loss", "Loss (CE + λ·DisCo)"),
    ("acc", "Accuracy"),
    ("eff_at_1pct", "eff@1%bkg"),
    ("jsd", "JSD"),
]


def _load_scalars(tb_dir: str, tags: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load scalar series for each tag. Returns {tag: (steps, values)}."""
    ea = event_accumulator.EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for t in tags:
        if t not in available:
            continue
        events = ea.Scalars(t)
        out[t] = (
            np.array([e.step for e in events], dtype=float),
            np.array([e.value for e in events], dtype=float),
        )
    return out


def _collect(run_dir: str) -> list[tuple[float, str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    """Gather (lambda, folder_name, {tag: (steps, values)}) sorted by lambda."""
    rows: list[tuple[float, str, dict]] = []
    for name in sorted(os.listdir(run_dir)):
        if not name.startswith("lambda_"):
            continue
        sub = os.path.join(run_dir, name)
        tb = os.path.join(sub, "tb")
        if not os.path.isdir(tb):
            continue
        try:
            lam = float(name.replace("lambda_", ""))
        except ValueError:
            continue
        tags = [f"{phase}/{m}" for phase in ("train", "val") for m, _ in METRICS]
        series = _load_scalars(tb, tags)
        if not series:
            logger.warning("No scalar data in %s", tb)
            continue
        rows.append((lam, name, series))
    rows.sort(key=lambda r: r[0])
    return rows


def _colors(lambdas: list[float]):
    """Return (color_list, ScalarMappable) using a log scale over lambda."""
    cmap = plt.get_cmap("viridis")
    shifted = np.array([max(l, 1e-3) for l in lambdas])
    norm = mcolors.LogNorm(vmin=shifted.min(), vmax=max(shifted.max(), shifted.min() * 10))
    colors = [cmap(norm(s)) for s in shifted]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return colors, sm


def _plot_one(ax, rows, colors, metric: str, phase: str) -> None:
    tag = f"{phase}/{metric}"
    for (lam, _name, series), c in zip(rows, colors):
        if tag not in series:
            continue
        steps, vals = series[tag]
        ax.plot(steps, vals, color=c, linewidth=1.2, alpha=0.9)


def plot_per_lambda(run_dir: str, split: bool) -> None:
    """Save one figure per lambda inside its lambda_<X> directory."""
    rows = _collect(run_dir)
    if not rows:
        raise SystemExit(f"No lambda_*/tb directories found under {run_dir}")

    ncols = 2 if not split else 4
    for lam, folder, series in rows:
        fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8), squeeze=False)
        for i, (metric, ylabel) in enumerate(METRICS):
            r, c = divmod(i, 2)
            if split:
                ax_t = axes[r][2 * c]
                ax_v = axes[r][2 * c + 1]
                for ax, phase in ((ax_t, "train"), (ax_v, "val")):
                    tag = f"{phase}/{metric}"
                    if tag in series:
                        st, vals = series[tag]
                        ax.plot(st, vals, color="C0", linewidth=1.5)
                    ax.set_title(f"{phase} {ylabel}")
                    ax.set_xlabel("epoch")
                    ax.set_ylabel(ylabel)
                    ax.grid(True, alpha=0.3)
            else:
                ax = axes[r][c]
                for phase, style in (("train", {"linestyle": "--", "linewidth": 1.2, "alpha": 0.7, "label": "train"}),
                                     ("val", {"linestyle": "-", "linewidth": 1.8, "alpha": 1.0, "label": "val"})):
                    tag = f"{phase}/{metric}"
                    if tag in series:
                        st, vals = series[tag]
                        ax.plot(st, vals, color="C0" if phase == "val" else "C1", **style)
                ax.set_title(ylabel)
                ax.set_xlabel("epoch")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best", fontsize=8)
                if metric == "loss":
                    ax.set_yscale("log")
        fig.suptitle(f"Learning curves: λ = {lam}", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        out_path = os.path.join(run_dir, folder, "learning_curves.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", out_path)


def plot_learning_curves(run_dir: str, out_path: str, split: bool) -> None:
    rows = _collect(run_dir)
    if not rows:
        raise SystemExit(f"No lambda_*/tb directories found under {run_dir}")
    lambdas = [r[0] for r in rows]
    colors, sm = _colors(lambdas)
    # Tuples are (lambda, folder, series); keep as-is for iteration below.

    ncols = 2 if not split else 4
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8), squeeze=False)

    for i, (metric, ylabel) in enumerate(METRICS):
        r, c = divmod(i, 2)
        if split:
            ax_t = axes[r][2 * c]
            ax_v = axes[r][2 * c + 1]
            _plot_one(ax_t, rows, colors, metric, "train")
            _plot_one(ax_v, rows, colors, metric, "val")
            ax_t.set_title(f"train {ylabel}")
            ax_v.set_title(f"val {ylabel}")
            for ax in (ax_t, ax_v):
                ax.set_xlabel("epoch")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
        else:
            ax = axes[r][c]
            for (lam, _name, series), col in zip(rows, colors):
                vtag = f"val/{metric}"
                ttag = f"train/{metric}"
                if ttag in series:
                    st, vt = series[ttag]
                    ax.plot(st, vt, color=col, linewidth=0.8, alpha=0.5, linestyle="--")
                if vtag in series:
                    sv, vv = series[vtag]
                    ax.plot(sv, vv, color=col, linewidth=1.4, alpha=0.95)
            ax.set_title(ylabel)
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if metric == "loss":
                ax.set_yscale("log")

    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.02, shrink=0.9)
    cbar.set_label("disco_lambda (log scale)")

    suffix = " (train dashed, val solid)" if not split else ""
    fig.suptitle(f"Learning curves: {os.path.basename(os.path.abspath(run_dir))}{suffix}", y=0.995)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s (n_lambdas=%d)", out_path, len(rows))


def main():
    p = argparse.ArgumentParser(description="Plot learning curves across a lambda sweep")
    p.add_argument("run_dir", help="Sweep directory containing lambda_*/tb subfolders")
    p.add_argument("--out", default=None, help="Output image path (default: <run_dir>/learning_curves.png)")
    p.add_argument("--split", action="store_true", help="Show train and val in separate panels")
    p.add_argument("--per-lambda", action="store_true",
                   help="Save a separate figure in each lambda_<X>/learning_curves.png")
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if args.per_lambda:
        plot_per_lambda(run_dir, split=args.split)
    else:
        out = args.out or os.path.join(run_dir, "learning_curves.png")
        plot_learning_curves(run_dir, out, split=args.split)


if __name__ == "__main__":
    main()
