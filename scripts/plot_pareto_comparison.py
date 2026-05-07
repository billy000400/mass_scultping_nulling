#!/usr/bin/env python3
"""Plot cure_disco_pareto and disco_pareto boundaries together with connecting lines."""

import csv
import os
import matplotlib.pyplot as plt


def load_summary(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            row["eff_at_1pct"] = float(row["eff_at_1pct"])
            row["jsd"] = float(row["jsd"])
            row["disco_lambda"] = float(row["disco_lambda"])
            rows.append(row)
    return sorted(rows, key=lambda r: r["jsd"])


def _plot(
    ax,
    cure: list[dict],
    disco: list[dict],
    *,
    title_suffix: str = "",
) -> None:
    jsd_c = [r["jsd"] for r in cure]
    eff_c = [r["eff_at_1pct"] for r in cure]
    jsd_d = [r["jsd"] for r in disco]
    eff_d = [r["eff_at_1pct"] for r in disco]

    ax.plot(jsd_c, eff_c, "o-", color="C0", linewidth=2, markersize=8, label="CURE+DisCo")
    ax.plot(jsd_d, eff_d, "s-", color="C1", linewidth=2, markersize=8, label="DisCo only")

    ax.set_xlabel("JSD (lower is better)")
    ax.set_ylabel("eff@1%bkg (higher is better)")
    title = "Pareto boundary: CURE+DisCo vs DisCo only"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)


def _save_fig(fig, base: str) -> None:
    os.makedirs(os.path.dirname(base), exist_ok=True)
    out_png = base + ".png"
    out_pdf = base + ".pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


def main():
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cure_path = os.path.join(proj, "runs/cure_disco_pareto/summary.csv")
    disco_path = os.path.join(proj, "runs/disco_pareto/summary.csv")

    cure = load_summary(cure_path)
    disco = load_summary(disco_path)

    runs_dir = os.path.join(proj, "runs")

    fig, ax = plt.subplots(figsize=(9, 6))
    _plot(ax, cure, disco)
    _save_fig(fig, os.path.join(runs_dir, "pareto_comparison"))

    max_jsd = 0.024
    cure_f = [r for r in cure if r["jsd"] < max_jsd]
    disco_f = [r for r in disco if r["jsd"] < max_jsd]
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    _plot(ax2, cure_f, disco_f, title_suffix=f"JSD < {max_jsd}")
    ax2.set_xlim(left=0, right=max_jsd)
    _save_fig(fig2, os.path.join(runs_dir, "pareto_comparison_jsd_lt_0.024"))


if __name__ == "__main__":
    main()
