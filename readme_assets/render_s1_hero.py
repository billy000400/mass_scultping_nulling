#!/usr/bin/env python3
"""Hero figure for S1 latent-edit method — README cover plate.

Design: "Orthogonal Silence" (see s1_hero_philosophy.md).

A single composition built around the geometry of subtraction.
The visual conceit is mass-conditional mean shift:
the embedding cloud is *displaced* along two directions v_1, v_2
that S1 identifies; the projection removes those displacements
and returns the cloud to its origin.

Outputs (written alongside this script):
  readme_assets/s1_hero.pdf
  readme_assets/s1_hero.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.patches import Circle, Rectangle


# ----- fonts ----------------------------------------------------------
FONT_DIR = (
    Path(__file__).resolve().parent.parent
    / ".claude" / "skills" / "canvas-design" / "canvas-fonts"
)
for fp in [
    FONT_DIR / "GeistMono-Regular.ttf",
    FONT_DIR / "GeistMono-Bold.ttf",
    FONT_DIR / "IBMPlexMono-Regular.ttf",
    FONT_DIR / "IBMPlexMono-Bold.ttf",
    FONT_DIR / "BricolageGrotesque-Regular.ttf",
    FONT_DIR / "BricolageGrotesque-Bold.ttf",
]:
    if fp.exists():
        fm.fontManager.addfont(str(fp))

FONT_MONO    = "Geist Mono"
FONT_DISPLAY = "Bricolage Grotesque"

mpl.rcParams.update({
    "mathtext.fontset": "cm",
    "mathtext.default": "regular",
})


# ----- palette --------------------------------------------------------
BG     = "#EFEAE0"   # bone
INK    = "#1A2A47"   # deep cobalt
ACCENT = "#B14238"   # cinnabar
SLATE  = "#5A6470"   # slate
GHOST  = "#C2BBA8"   # faded ground

mpl.rcParams.update({
    "savefig.facecolor": BG,
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
})


# ----- helpers --------------------------------------------------------
def hline(ax, y, x0, x1, lw=0.6, color=INK, alpha=1.0):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, alpha=alpha,
            solid_capstyle="butt", zorder=1)


def tick_row(ax, y, x0, x1, n, lw=0.45, color=INK, h=0.06):
    xs = np.linspace(x0, x1, n)
    for x in xs:
        ax.plot([x, x], [y, y + h], color=color, lw=lw,
                solid_capstyle="butt", zorder=2)


def label(ax, x, y, txt, *, size=6.5, family=FONT_MONO,
          color=INK, ha="left", va="bottom", weight="normal",
          alpha=1.0, rotation=0):
    ax.text(x, y, txt, fontsize=size, family=family, color=color,
            ha=ha, va=va, weight=weight, alpha=alpha, rotation=rotation,
            zorder=10)


def arrow(ax, x0, y0, x1, y1, color=INK, lw=0.7,
          head_width=0.10, head_length=0.14):
    """Slim line arrow with a discrete triangular head."""
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy)
    if L < 1e-6:
        return
    ux, uy = dx / L, dy / L
    # body stops just before the head
    body_end_x = x1 - ux * head_length * 0.9
    body_end_y = y1 - uy * head_length * 0.9
    ax.plot([x0, body_end_x], [y0, body_end_y],
            color=color, lw=lw, solid_capstyle="butt", zorder=4)
    # head triangle
    px, py = -uy, ux
    bx = x1 - ux * head_length
    by = y1 - uy * head_length
    p1 = (bx + px * head_width/2, by + py * head_width/2)
    p2 = (bx - px * head_width/2, by - py * head_width/2)
    ax.fill([x1, p1[0], p2[0]], [y1, p1[1], p2[1]],
            color=color, lw=0, zorder=4)


# ----- main canvas ----------------------------------------------------
def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    W, H = 13.0, 8.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    M = 0.6
    ax.add_patch(Rectangle((M, M), W - 2*M, H - 2*M,
                           fill=False, ec=INK, lw=0.5, zorder=3))
    tick_row(ax, H - M, M, W - M, n=41)
    tick_row(ax, M - 0.06, M, W - M, n=41)

    # ============================================================
    # HEADER
    # ============================================================
    y_head = H - M - 0.35
    label(ax, M + 0.4, y_head + 0.20,
          "S1  ·  Latent Subspace Editing",
          size=13.5, family=FONT_DISPLAY, weight="bold")
    label(ax, W - M - 0.4, y_head + 0.20,
          "MASS - DECORRELATED   JET   TAGGING",
          size=8.0, family=FONT_MONO, color=SLATE, ha="right")
    hline(ax, y_head - 0.06, M + 0.4, W - M - 0.4, lw=0.5)
    label(ax, M + 0.4, y_head - 0.30,
          "QCD-only  ·  class-conditional  ·  mass-binned SVD  ·  closed form",
          size=7.2, family=FONT_MONO, color=SLATE)
    label(ax, W - M - 0.4, y_head - 0.30,
          r"rev  ·  S1 / k=2 / $\alpha$=1",
          size=7.2, family=FONT_MONO, color=SLATE, ha="right")

    # ============================================================
    # MAIN PIPELINE
    # ============================================================
    y_band = 4.55
    radius = 1.55
    cx_L = 2.65
    cx_R = 10.35
    # The middle is "empty" — only the projection arrow lives there,
    # and the equation goes in the band below.

    # --- visual conceit: mean-shift cloud --------------------------
    rng = np.random.default_rng(11)
    n_pts = 260
    theta = rng.uniform(0, 2*np.pi, n_pts)
    rho = np.sqrt(rng.uniform(0, 1, n_pts)) * 0.55
    # base cloud (homogeneous disk) — the noise / within-bin variance
    base = np.stack([rho*np.cos(theta), rho*np.sin(theta)], axis=1)

    # mass parameter per point in [-1, 1] (analogous to centred m_J)
    m_per_pt = rng.uniform(-1.0, 1.0, n_pts)

    # basis directions
    v1 = np.array([1.0, 0.0])
    raw_v2 = np.array([np.cos(np.deg2rad(58)), np.sin(np.deg2rad(58))])
    v2 = raw_v2 - (raw_v2 @ v1) * v1
    v2 = v2 / np.linalg.norm(v2)

    # mean-shift component along v1 (dominant) and v2 (subdominant)
    shift = (m_per_pt[:, None] * 0.68) * v1[None, :] \
          + (m_per_pt[:, None] * 0.35) * v2[None, :]

    cloud_before = base + shift     # cloud has a coherent stretch along v1+v2
    cloud_after  = base             # mean shift removed → centred ball

    # ---- LEFT panel: x ∈ R^128 with the two mass-shift axes -----
    ax.add_patch(Circle((cx_L, y_band), radius,
                        fill=False, ec=INK, lw=0.6, zorder=3))
    pts = cloud_before * radius * 0.92
    # colour points by their mass parameter to show the shift visually
    ax.scatter(cx_L + pts[:, 0], y_band + pts[:, 1],
               c=m_per_pt, cmap="RdBu_r", vmin=-1.1, vmax=1.1,
               s=5.0, alpha=0.85, linewidths=0, zorder=4)
    # crosshair (faint)
    ax.plot([cx_L - radius*1.10, cx_L + radius*1.10], [y_band, y_band],
            color=INK, lw=0.3, alpha=0.20, zorder=2)
    ax.plot([cx_L, cx_L], [y_band - radius*1.10, y_band + radius*1.10],
            color=INK, lw=0.3, alpha=0.20, zorder=2)
    # mass-shift arrows (v1 horizontal, v2 diagonal)
    LA = radius * 1.00
    # v1 — through origin
    arrow(ax, cx_L - v1[0]*LA*0.45, y_band - v1[1]*LA*0.45,
          cx_L + v1[0]*LA*0.92, y_band + v1[1]*LA*0.92,
          color=ACCENT, lw=1.3, head_width=0.13, head_length=0.18)
    arrow(ax, cx_L - v2[0]*LA*0.45, y_band - v2[1]*LA*0.45,
          cx_L + v2[0]*LA*0.92, y_band + v2[1]*LA*0.92,
          color=ACCENT, lw=1.3, head_width=0.13, head_length=0.18)
    # labels at the actual arrow tips, offset just past them
    pad = 0.18
    label(ax, cx_L + v1[0]*(LA*0.92) + pad, y_band + v1[1]*(LA*0.92) - 0.02,
          r"$v_{1}$", size=11.5, family="serif", color=ACCENT,
          weight="bold", ha="left", va="center")
    label(ax, cx_L + v2[0]*(LA*0.92) + 0.04,
          y_band + v2[1]*(LA*0.92) + 0.18,
          r"$v_{2}$", size=11.5, family="serif", color=ACCENT,
          weight="bold", ha="left", va="bottom")
    # captions under the panel
    label(ax, cx_L, y_band - radius - 0.32,
          r"$\mathbf{x}\ \in\ \mathbb{R}^{128}$",
          size=12, family="serif", ha="center", weight="bold")
    label(ax, cx_L, y_band - radius - 0.60,
          "CLS  EMBEDDING  ·  ParT (FROZEN)",
          size=6.8, family=FONT_MONO, color=SLATE, ha="center")

    # ---- MIDDLE: projection arrow ------------------------------
    mid_x0 = cx_L + radius + 0.30
    mid_x1 = cx_R - radius - 0.30
    arrow_y = y_band
    arrow(ax, mid_x0, arrow_y, mid_x1, arrow_y,
          color=INK, lw=0.9, head_width=0.18, head_length=0.30)
    # label above the arrow
    mid_cx = (mid_x0 + mid_x1) / 2
    label(ax, mid_cx, arrow_y + 0.45,
          r"$\;-\;\alpha\,V^{\top}V\,(x-\mu)$",
          size=18, family="serif", color=INK, ha="center", va="center",
          weight="bold")
    # one-line subtitle below the arrow
    label(ax, mid_cx, arrow_y - 0.45,
          "ORTHOGONAL  PROJECTION   ·   rank  k = 2   on   the   S1   basis",
          size=7.6, family=FONT_MONO, color=SLATE, ha="center")

    # ---- RIGHT panel: edited embedding x̃ ----------------------
    ax.add_patch(Circle((cx_R, y_band), radius,
                        fill=False, ec=INK, lw=0.6, zorder=3))
    pts_a = cloud_after * radius * 0.92
    ax.scatter(cx_R + pts_a[:, 0], y_band + pts_a[:, 1],
               c=m_per_pt, cmap="RdBu_r", vmin=-1.1, vmax=1.1,
               s=5.0, alpha=0.85, linewidths=0, zorder=4)
    ax.plot([cx_R - radius*1.10, cx_R + radius*1.10], [y_band, y_band],
            color=INK, lw=0.3, alpha=0.20, zorder=2)
    ax.plot([cx_R, cx_R], [y_band - radius*1.10, y_band + radius*1.10],
            color=INK, lw=0.3, alpha=0.20, zorder=2)
    # ghost arrows where v1/v2 used to be (visible absence)
    arrow(ax, cx_R - v1[0]*LA*0.45, y_band - v1[1]*LA*0.45,
          cx_R + v1[0]*LA*0.92, y_band + v1[1]*LA*0.92,
          color=GHOST, lw=1.0, head_width=0.10, head_length=0.16)
    arrow(ax, cx_R - v2[0]*LA*0.45, y_band - v2[1]*LA*0.45,
          cx_R + v2[0]*LA*0.92, y_band + v2[1]*LA*0.92,
          color=GHOST, lw=1.0, head_width=0.10, head_length=0.16)
    label(ax, cx_R, y_band - radius - 0.32,
          r"$\tilde{\mathbf{x}}\ \in\ \mathbb{R}^{128}$",
          size=12, family="serif", ha="center", weight="bold")
    label(ax, cx_R, y_band - radius - 0.60,
          "EDITED  CLS  EMBEDDING  ·  HEAD  →  $T_{Xbb}$",
          size=6.8, family=FONT_MONO, color=SLATE, ha="center")

    # ============================================================
    # EQUATION BAND
    # ============================================================
    eq_y = 2.55
    label(ax, W/2, eq_y,
          r"$\tilde{\mathbf{x}}\;=\;\mathbf{x}\;-\;\alpha\,V^{\top}V\,(\mathbf{x}-\mu)$",
          size=22, family="serif", weight="bold",
          ha="center", va="center")
    hline(ax, eq_y - 0.48, W/2 - 3.8, W/2 + 3.8, lw=0.4, color=SLATE)

    # ============================================================
    # SCREE SPECTRUM BAND  (raised to avoid tick-label collision
    # with the footer)
    # ============================================================
    scree_y_top = 1.65
    scree_y_base = 1.20
    scree_x0 = M + 0.45
    scree_x1 = W - M - 0.45
    label(ax, scree_x0, scree_y_top + 0.10,
          r"S1  BASIS   ·   $\sigma_{i}/\sigma_{1}$  on  log  scale   ·   "
          r"20  quantile  $m_J$  bins   ·   centred  class-mean  SVD",
          size=7.4, family=FONT_MONO, color=INK)
    label(ax, scree_x1, scree_y_top + 0.10,
          "k = 2  directions  removed",
          size=7.4, family=FONT_MONO, color=ACCENT,
          ha="right", weight="bold")

    sigs = np.array([1.0, 0.681, 0.180, 0.045, 0.026, 0.020, 0.013,
                     0.008, 0.006, 0.002, 0.0011, 0.0007, 0.0003,
                     1.6e-4, 1.2e-4, 1.0e-4, 1.3e-5])
    n_sig = len(sigs)
    log_sig = np.clip(np.log10(sigs), -6.0, 0.0)
    norm = (log_sig - log_sig.min()) / (log_sig.max() - log_sig.min())
    bar_h_max = scree_y_top - scree_y_base - 0.10
    bar_h = norm * bar_h_max

    bar_pitch = (scree_x1 - scree_x0 - 0.6) / (n_sig - 1)
    x0_bars = scree_x0 + 0.30
    for i, h in enumerate(bar_h):
        x = x0_bars + i * bar_pitch
        col = ACCENT if i < 2 else INK
        lw  = 1.6 if i < 2 else 0.55
        ax.plot([x, x], [scree_y_base, scree_y_base + max(h, 0.05)],
                color=col, lw=lw, solid_capstyle="butt", zorder=4)
        ax.plot([x, x], [scree_y_base - 0.05, scree_y_base],
                color=INK, lw=0.4, solid_capstyle="butt")
        if i in (0, 1, 2, 4, 8, 16):
            label(ax, x, scree_y_base - 0.20, f"{i+1}",
                  size=6.2, family=FONT_MONO, color=SLATE,
                  ha="center", va="top")
    hline(ax, scree_y_base, scree_x0, scree_x1, lw=0.35, color=INK)

    # Lower caption row deliberately omitted — the header above the
    # scree already names the spectrum; the tick numerals carry the rest.

    # ============================================================
    # FOOTER
    # ============================================================
    foot_y = M + 0.15
    hline(ax, foot_y + 0.22, M + 0.4, W - M - 0.4, lw=0.4, color=SLATE)
    label(ax, M + 0.4, foot_y,
          "MASS - SCULPTING  NULLING   ·   S 1   LATENT   EDIT",
          size=6.8, family=FONT_MONO, color=SLATE)
    label(ax, W - M - 0.4, foot_y,
          "PLATE   ·   001",
          size=6.8, family=FONT_MONO, color=SLATE, ha="right")

    # ---- save outputs ------------------------------------------
    pdf = out_dir / "s1_hero.pdf"
    png = out_dir / "s1_hero.png"
    fig.savefig(pdf, bbox_inches=None, pad_inches=0, facecolor=BG)
    fig.savefig(png, dpi=240, bbox_inches=None, pad_inches=0, facecolor=BG)
    plt.close(fig)
    print(f"saved {pdf}")
    print(f"saved {png}")


if __name__ == "__main__":
    main()
