#!/usr/bin/env python3
"""
BearMamba-3 architecture diagram (Figure 1).

Two-row layout:
  Top row: Input → Conv Embed → Mamba-3 Block ×N → Mean Pool → Classifier → CE Loss
  Bottom row (L_kin branch): angle activations → f_inst → min|f-f_fault| → L_kin

Additional annotation boxes:
  - Multi-sensor: n_s input channels → same Conv Embed
  - Physical targets: BPFO / BPFI / BSF (per-sample RPM)
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pathlib

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
})

OUT_DIR = pathlib.Path("/home/jeffwork/论文8/results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── increased height to avoid top/bottom overlaps ────────────────────────────
fig, ax = plt.subplots(figsize=(11.0, 5.8))
ax.set_xlim(0, 11); ax.set_ylim(0, 6.2)
ax.axis("off")

# ── helpers ──────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel=None,
        fc="#dce6f1", ec="#336699", lw=1.4, fontsize=8.5, bold=False):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle="round,pad=0.08",
                        fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(b)
    weight = "bold" if bold else "normal"
    ax.text(x, y + (0.1 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight=weight, zorder=4)
    if sublabel:
        ax.text(x, y - 0.22, sublabel,
                ha="center", va="center", fontsize=6.5,
                color="#666666", zorder=4)

def arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.2, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, shrinkA=3, shrinkB=3),
                zorder=5)

def dashed_arrow(ax, x0, y0, x1, y1, color="#aa5500", lw=1.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                linestyle="dashed", shrinkA=3, shrinkB=3),
                zorder=5)


# ── MAIN PATH ────────────────────────────────────────────────────────────────
Y_MAIN = 3.8   # y of main path

# Input box (multi-sensor)
box(ax, 0.85, Y_MAIN, 1.2, 0.75,
    r"Input $\mathbf{x}$",
    r"$T \times n_s$",
    fc="#e8f5e9", ec="#2e7d32")

arrow(ax, 1.45, Y_MAIN, 1.85, Y_MAIN)

# Conv Embed
box(ax, 2.35, Y_MAIN, 0.95, 0.75,
    "Conv\nEmbed",
    r"$k=3,s=2$",
    fc="#e3f2fd", ec="#1565c0")

arrow(ax, 2.83, Y_MAIN, 3.15, Y_MAIN)

# Mamba-3 block x4
box(ax, 3.75, Y_MAIN, 1.20, 0.80,
    "Mamba-3\nBlock",
    r"$\times N_\ell = 4$",
    fc="#ede7f6", ec="#6a1b9a", bold=True)

arrow(ax, 4.35, Y_MAIN, 4.75, Y_MAIN)

# Mean Pool + LN
box(ax, 5.20, Y_MAIN, 0.85, 0.75,
    "Mean\nPool + LN",
    r"$\in\mathbb{R}^{d}$",
    fc="#e3f2fd", ec="#1565c0")

arrow(ax, 5.62, Y_MAIN, 6.02, Y_MAIN)

# Classifier
box(ax, 6.50, Y_MAIN, 0.90, 0.75,
    "Linear\nClassifier",
    r"$d \to C$",
    fc="#fff8e1", ec="#f57f17")

arrow(ax, 6.95, Y_MAIN, 7.35, Y_MAIN)

# CE Loss
box(ax, 7.85, Y_MAIN, 0.90, 0.75,
    r"$\mathcal{L}_{\mathrm{CE}}$",
    None,
    fc="#fce4ec", ec="#b71c1c", lw=1.6, bold=True)

# Total loss combiner
box(ax, 9.10, Y_MAIN, 1.00, 0.85,
    r"$\mathcal{L}_{\mathrm{total}}$",
    r"$= \mathcal{L}_{CE} + \lambda\mathcal{L}_{kin}$",
    fc="#fbe9e7", ec="#bf360c", lw=1.8, bold=True)

arrow(ax, 8.30, Y_MAIN, 8.60, Y_MAIN)


# ── L_KIN BRANCH ─────────────────────────────────────────────────────────────
Y_KIN = 1.9   # y of L_kin branch
Y_HOOK = 2.60  # from Mamba-3 block downward tap

# Mamba-3 → L_kin branch (vertical tap)
dashed_arrow(ax, 3.75, Y_MAIN - 0.40, 3.75, Y_HOOK + 0.1)
dashed_arrow(ax, 3.75, Y_HOOK, 3.75, Y_KIN + 0.38)

# angle activations box
box(ax, 3.75, Y_KIN, 1.10, 0.70,
    r"$\theta_{h,k}(t)$",
    "angle activations",
    fc="#f3e5f5", ec="#880e4f", lw=1.2)

dashed_arrow(ax, 4.30, Y_KIN, 4.80, Y_KIN)

# f_inst
box(ax, 5.30, Y_KIN, 1.15, 0.70,
    r"$f_{h,k}(t)$",
    r"$=\tanh(\theta)\cdot\Delta t\cdot f_s^{\rm eff}/2$",
    fc="#f3e5f5", ec="#880e4f", lw=1.2)

dashed_arrow(ax, 5.88, Y_KIN, 6.30, Y_KIN)

# min|f - f_fault|
box(ax, 6.95, Y_KIN, 1.20, 0.70,
    r"$\min_{h,k}|f_{h,k}-f_c|$",
    r"cover over $\mathcal{F}_i$",
    fc="#f3e5f5", ec="#880e4f", lw=1.2)

dashed_arrow(ax, 7.55, Y_KIN, 7.90, Y_KIN)

# L_kin box
box(ax, 8.40, Y_KIN, 0.90, 0.70,
    r"$\mathcal{L}_{\mathrm{kin}}$",
    r"$\lambda = 0.01$",
    fc="#fce4ec", ec="#880e4f", lw=1.6, bold=True)

# L_kin → total loss
dashed_arrow(ax, 8.85, Y_KIN, 9.10, Y_MAIN - 0.43)

# CE → total loss (re-draw to avoid duplication)
arrow(ax, 8.30, Y_MAIN, 8.60, Y_MAIN, color="#555555")


# ── Fault target annotation ───────────────────────────────────────────────────
# Physical targets box — directly below the min|f-f_fault| box for a clean arrow
ft_x, ft_y = 6.95, 0.65
box(ax, ft_x, ft_y, 2.30, 0.80,
    r"$\mathcal{F}_i$ (per-sample RPM)",
    r"$f_{\mathrm{BPFO}},f_{\mathrm{BPFI}},f_{\mathrm{BSF}}$, harmonics (15 targets)",
    fc="#e8f5e9", ec="#2e7d32", lw=1.0, fontsize=7.5)

# single upward arrow: fault targets → cover computation
dashed_arrow(ax, 6.95, ft_y + 0.40, 6.95, Y_KIN - 0.35,
             color="#2e7d32")


# ── Multi-sensor annotation (left side, between rows) ────────────────────────
ms_x, ms_y = 0.85, Y_MAIN - 1.10
ax.text(ms_x, ms_y,
        r"$n_s = 1$: DE only" + "\n"
        + r"$n_s = 2$: DE + FE (CWRU)" + "\n"
        + r"$n_s = 2$: H + V (XJTU-SY)",
        ha="center", va="top", fontsize=7,
        color="#2e7d32",
        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#2e7d32", alpha=0.9))


# ── L_kin DE-channel note — moved to left, below multi-sensor box ─────────────
ax.text(0.85, 1.35,
        r"$\mathcal{L}_{\mathrm{kin}}$: DE channel only" + "\n"
        + "(FE geometry differs)",
        ha="center", va="top", fontsize=7, color="#880e4f",
        bbox=dict(boxstyle="round,pad=0.3", fc="#f3e5f5", ec="#880e4f", alpha=0.85))


# ── MIMO note — raised above dimension labels to avoid overlap ────────────────
ax.text(3.75, Y_MAIN + 1.10,
        "SISO (is_mimo=False)\nMIMO backward: sm_86 hw limit",
        ha="center", va="bottom", fontsize=6.5, color="#6a1b9a",
        bbox=dict(boxstyle="round,pad=0.2", fc="#ede7f6", ec="#6a1b9a", alpha=0.8))


# ── Section title ─────────────────────────────────────────────────────────────
ax.text(5.5, 6.05,
        "BearMamba-3 Architecture (training graph, inference drops L_kin)",
        ha="center", va="top", fontsize=10, fontweight="bold", color="#222222")

# Legend
leg_items = [
    mpatches.Patch(fc="#e8f5e9",  ec="#2e7d32",  label="Input / outputs"),
    mpatches.Patch(fc="#e3f2fd",  ec="#1565c0",  label="Embedding / pooling"),
    mpatches.Patch(fc="#ede7f6",  ec="#6a1b9a",  label="Mamba-3 backbone"),
    mpatches.Patch(fc="#f3e5f5",  ec="#880e4f",  label="L_kin branch (train only)"),
    mpatches.Patch(fc="#fce4ec",  ec="#b71c1c",  label="Loss functions"),
]
ax.legend(handles=leg_items, loc="lower right",
          fontsize=7.0, framealpha=0.9,
          bbox_to_anchor=(1.0, 0.0))


# ── dimension annotations on main path ───────────────────────────────────────
# Placed BELOW the MIMO note (MIMO note top ≈ Y_MAIN+1.10, dim labels at Y_MAIN+0.55)
for x, label in [
    (1.45, r"$T\!\times\!n_s$"),
    (2.83, r"$L\!\times\!d$"),
    (4.35, r"$L\!\times\!d$"),
    (5.62, r"$d$"),
    (6.95, r"$C$"),
]:
    ax.text(x, Y_MAIN + 0.55, label, ha="center", va="bottom",
            fontsize=7.0, color="#1565c0")


fig.tight_layout()
out_stem = "arch_bearmamba3"
fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=200)
print(f"Saved → {OUT_DIR / out_stem}.{{pdf,png}}")
plt.close(fig)
