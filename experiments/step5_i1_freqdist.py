#!/usr/bin/env python3
"""
Step 5 / I1: Instantaneous frequency distribution vs fault frequency spectral lines.

Figure layout (2-panel, 7x3.5 in):
  (a) KDE of f_bar at ep=1 vs ep=50 — shows L_kin shifts state distribution
      toward fault frequencies
  (b) Coverage convergence — nearest-state distance per fault frequency over epochs

Data source: results/exp02_snr0_kin/ (CWRU 4-class, SNR=0dB, λ=0.01, 5 seeds)

Usage:
    source venv/bin/activate
    python experiments/step5_i1_freqdist.py
"""
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ─── paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path("/home/jeffwork/论文8")
SRC_DIR = ROOT / "results" / "exp02_snr0_kin"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── fault frequency layout  (col i*3 … i*3+2 = harmonics 1x,2x,3x) ─────────
FAULT_NAMES = ["fr", "FTF", "BPFO", "2BSF", "BPFI"]
FAULT_COLORS = {
    "fr":   "#808080",
    "FTF":  "#808080",
    "BPFO": "#1f77b4",
    "2BSF": "#ff7f0e",
    "BPFI": "#2ca02c",
}
# Target 1st-harmonic values at ~1797 rpm (CWRU SKF 6205)
FAULT_1X = {"fr": 29.93, "FTF": 11.92, "BPFO": 107.30, "2BSF": 141.09, "BPFI": 162.10}

PLOT_FAULTS = ["BPFO", "2BSF", "BPFI"]   # fr and FTF already covered at ep=1
EPOCH_INTEREST = [1, 10, 30, 50]
FREQ_XLIM = (0, 350)
ALL_EPOCHS = [1, 10, 20, 30, 40, 50]


def load_fbar_for_epoch(epoch: int) -> np.ndarray:
    """Return flattened f_bar (Hz) array for all seeds at the given epoch."""
    files = sorted(SRC_DIR.glob(f"*ep{epoch:03d}*.npz"))
    if not files:
        raise FileNotFoundError(f"No snapshots for epoch {epoch} in {SRC_DIR}")
    return np.concatenate([np.load(f)["f_bar"].flatten() for f in files])


def coverage_distance_at_epoch(epoch: int) -> dict[str, float]:
    """For each fault type, return mean nearest-state distance (Hz) at given epoch."""
    files = sorted(SRC_DIR.glob(f"*ep{epoch:03d}*.npz"))
    per_fault: dict[str, list[float]] = {n: [] for n in PLOT_FAULTS}
    for f in files:
        d = np.load(f)
        f_bar   = d["f_bar"]    # (B, S)
        f_fault = d["f_fault"]  # (B, 15)
        B = f_bar.shape[0]
        for b in range(B):
            states = f_bar[b]
            for i, name in enumerate(FAULT_NAMES):
                if name not in PLOT_FAULTS:
                    continue
                freq1x = float(f_fault[b, i * 3])
                dist   = float(np.min(np.abs(states - freq1x)))
                per_fault[name].append(dist)
    return {name: float(np.mean(per_fault[name])) for name in PLOT_FAULTS}


# ─── collect data ─────────────────────────────────────────────────────────────
print("Loading snapshots …")
fbar_ep1  = load_fbar_for_epoch(1)
fbar_ep50 = load_fbar_for_epoch(50)

coverage = {}
for ep in ALL_EPOCHS:
    try:
        coverage[ep] = coverage_distance_at_epoch(ep)
    except FileNotFoundError:
        pass

print(f"  f_bar ep=1 : n={len(fbar_ep1)}, mean={fbar_ep1.mean():.1f}, "
      f"p90={np.percentile(fbar_ep1,90):.1f} Hz")
print(f"  f_bar ep=50: n={len(fbar_ep50)}, mean={fbar_ep50.mean():.1f}, "
      f"p90={np.percentile(fbar_ep50,90):.1f} Hz")
print("  Coverage distances (Hz):")
for ep in sorted(coverage):
    row = ", ".join(f"{n}:{coverage[ep][n]:.2f}" for n in PLOT_FAULTS)
    print(f"    ep={ep:3d}: {row}")


# ─── KDE helper ───────────────────────────────────────────────────────────────
def kde_curve(data: np.ndarray, x_min: float, x_max: float, n_pts: int = 512):
    clipped = data[(data >= x_min) & (data <= x_max)]
    if len(clipped) < 10:
        x = np.linspace(x_min, x_max, n_pts)
        return x, np.zeros(n_pts)
    bw = 0.15  # fraction of data range; tune for smoothness
    kde = gaussian_kde(clipped, bw_method=bw)
    x = np.linspace(x_min, x_max, n_pts)
    return x, kde(x)


# ─── pre-load all epoch data for panel (a) ────────────────────────────────────
import matplotlib.cm as cm
EPOCH_KDE = {}
for ep in EPOCH_INTEREST:
    try:
        fbar = load_fbar_for_epoch(ep)
        EPOCH_KDE[ep] = fbar
    except FileNotFoundError:
        pass

# ─── figure: 3-panel layout ──────────────────────────────────────────────────
fig = plt.figure(figsize=(10.5, 3.4))
gs  = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.0, 1.4], wspace=0.42)
ax_full = fig.add_subplot(gs[0])   # full distribution 0-350 Hz, multi-epoch
ax_zoom = fig.add_subplot(gs[1])   # zoomed fault-freq zone 75-220 Hz
ax_cov  = fig.add_subplot(gs[2])   # coverage convergence

axes = [ax_full, ax_zoom, ax_cov]  # for subsequent code

# ── panel (a): full distribution, multi-epoch gradient ───────────────────────
ax = ax_full
ax.set_title("(a) State Frequency Distribution")

x_min, x_max = FREQ_XLIM
cmap = cm.Blues
epoch_list = sorted(EPOCH_KDE.keys())
n_ep = len(epoch_list)

for idx, ep in enumerate(epoch_list):
    c_frac = 0.30 + 0.70 * (idx / max(n_ep - 1, 1))
    color  = cmap(c_frac)
    lw     = 0.9 + 0.8 * (idx / max(n_ep - 1, 1))
    ls     = "--" if ep == 1 else "-"
    alpha  = 0.55 + 0.45 * (idx / max(n_ep - 1, 1))
    fbar   = EPOCH_KDE[ep]
    xx, yy = kde_curve(fbar, x_min, x_max)
    ax.plot(xx, yy, color=color, lw=lw, ls=ls, alpha=alpha,
            label=f"ep {ep}")

ax.set_xlabel("Instantaneous Frequency (Hz)")
ax.set_ylabel("Density")
ax.set_xlim(x_min, x_max)
ax.set_ylim(bottom=0)
ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
ax.relim(); ax.autoscale_view(scalex=False)
y_top = ax.get_ylim()[1]

for name in PLOT_FAULTS:
    f0 = FAULT_1X[name]
    ax.axvline(f0, color=FAULT_COLORS[name], lw=1.1, ls=":", alpha=0.85)
    ax.text(f0 + 2, y_top * 0.04, name,
            color=FAULT_COLORS[name], fontsize=7.5, va="bottom", rotation=90)

# highlight fault-freq zone
ax.axvspan(75, 220, alpha=0.07, color="#888888", label="_zoom")
ax.text(147, y_top * 0.82, "(b)\nzoom", fontsize=7, ha="center",
        color="#666666", style="italic")

# ── panel (b): zoomed fault-frequency zone ───────────────────────────────────
ax = ax_zoom
ax.set_title("(b) Fault-Frequency Zone")

xi_min, xi_max = 75, 225
for idx, ep in enumerate(epoch_list):
    c_frac = 0.30 + 0.70 * (idx / max(n_ep - 1, 1))
    color  = cmap(c_frac)
    lw     = 0.9 + 0.8 * (idx / max(n_ep - 1, 1))
    ls     = "--" if ep == 1 else "-"
    alpha  = 0.65 + 0.35 * (idx / max(n_ep - 1, 1))
    fbar   = EPOCH_KDE[ep]
    xx, yy = kde_curve(fbar, xi_min, xi_max, n_pts=300)
    ax.plot(xx, yy, color=color, lw=lw, ls=ls, alpha=alpha)

ax.set_xlabel("Instantaneous Frequency (Hz)")
ax.set_ylabel("Density")
ax.set_xlim(xi_min, xi_max)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50))

# fault labels after ylim is settled
ax.relim(); ax.autoscale_view(scalex=False)
yi_top = ax.get_ylim()[1]
for name in PLOT_FAULTS:
    f0 = FAULT_1X[name]
    ax.axvline(f0, color=FAULT_COLORS[name], lw=1.1, ls=":", alpha=0.9)
    ax.text(f0 + 1.5, yi_top * 0.04, name,
            color=FAULT_COLORS[name], fontsize=7.5, va="bottom", rotation=90)

# ── panel (c): coverage convergence ──────────────────────────────────────────
ax2 = ax_cov
ax2.set_title("(c) Nearest-State Distance to Fault Frequency")

epochs_sorted = sorted(coverage.keys())
for name in PLOT_FAULTS:
    dists = [coverage[ep][name] for ep in epochs_sorted]
    ax2.plot(epochs_sorted, dists,
             marker="o", markersize=4, lw=1.5,
             color=FAULT_COLORS[name], label=name)

ax2.set_xlabel("Training Epoch")
ax2.set_ylabel("Nearest State Distance (Hz)")
ax2.set_xlim(0, 52)
ax2.set_xticks(ALL_EPOCHS)
ax2.set_ylim(bottom=0)
ax2.legend(loc="upper right", framealpha=0.85)
ax2.grid(axis="y", ls=":", alpha=0.4)

# ── finalize ─────────────────────────────────────────────────────────────────
fig.suptitle(
    "BearMamba-3 / CWRU SNR=0 dB / L_kin (λ=0.01, cover) / 5 seeds",
    fontsize=8, color="#444444",
)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_pdf = OUT_DIR / "i1_freq_distribution_cwru.pdf"
out_png = OUT_DIR / "i1_freq_distribution_cwru.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=200)
print(f"\nSaved → {out_pdf}")
print(f"Saved → {out_png}")
plt.close(fig)
