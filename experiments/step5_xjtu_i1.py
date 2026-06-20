#!/usr/bin/env python3
"""
Step 5 / XJTU I1 + CWRU comparison: why L_kin helps CWRU but not XJTU.

Key finding: XJTU run-to-failure signals naturally concentrate near fault
frequencies (BPFO coverage dist < 1 Hz at ep=1), so L_kin provides marginal
additional alignment. CWRU low-SNR signals start far from fault frequencies
(BPFO ~15 Hz at ep=1) → L_kin drives 5× convergence.

2-panel figure:
  (a) Coverage distance (median nearest-state dist) across training epochs
      CWRU SNR=0dB vs XJTU Cond3 — both with L_kin
  (b) Normalised KDE at ep=50: XJTU f_bar in 0-280 Hz zone
      (context: distribution shape with fault freq lines)
"""
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

ROOT     = pathlib.Path("/home/jeffwork/论文8/results")
CWRU_DIR = ROOT / "exp02_snr0_kin"
XJTU_DIR = ROOT / "exp_xjtu_lobo_kin"
OUT_DIR  = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CWRU (SKF 6205 @~1797rpm) fault frequencies (Hz)
CWRU_FAULTS = {"BPFO": 107.30, "2BSF": 141.09, "BPFI": 162.10}
# XJTU (@2400rpm Cond3) fault frequencies (Hz)
XJTU_FAULTS = {"BPFO": 123.32, "2BSF": 165.33, "BPFI": 196.68}

EPOCHS = [1, 10, 20, 30, 40, 50]
COLORS = {"BPFO": "#d62728", "2BSF": "#ff7f0e", "BPFI": "#9467bd"}


def coverage_distance_epoch(src_dir: pathlib.Path, epoch: int, target_freq: float) -> float:
    """Mean nearest-state distance to target_freq across all snapshots (consistent with I1 figure)."""
    fbar_parts = []
    for fpath in sorted(src_dir.glob(f"*ep{epoch:03d}*.npz")):
        fbar_parts.append(np.load(fpath)["f_bar"])  # (B, 256)
    if not fbar_parts:
        return np.nan
    fbar = np.concatenate(fbar_parts, axis=0)  # (N, 256)
    return float(np.abs(fbar - target_freq).min(axis=1).mean())


def kde_curve(src_dir: pathlib.Path, epoch: int, x_min: float, x_max: float, n_pts=512):
    fbar_parts = []
    for fpath in sorted(src_dir.glob(f"*ep{epoch:03d}*.npz")):
        fbar_parts.append(np.load(fpath)["f_bar"].ravel())
    if not fbar_parts:
        return None, None
    data = np.concatenate(fbar_parts)
    data_ok = data[(data >= x_min) & (data <= x_max)]
    if len(data_ok) < 50:
        return None, None
    bw = data_ok.std() * len(data_ok) ** (-0.2) * 0.6
    kde = gaussian_kde(data_ok, bw_method=bw / data_ok.std())
    xs = np.linspace(x_min, x_max, n_pts)
    return xs, kde(xs)


# ── compute data ──────────────────────────────────────────────────────────────
print("Computing coverage distances …")
cwru_cov = {n: [] for n in CWRU_FAULTS}
xjtu_cov = {n: [] for n in XJTU_FAULTS}
for ep in EPOCHS:
    for name, freq in CWRU_FAULTS.items():
        cwru_cov[name].append(coverage_distance_epoch(CWRU_DIR, ep, freq))
    for name, freq in XJTU_FAULTS.items():
        xjtu_cov[name].append(coverage_distance_epoch(XJTU_DIR, ep, freq))

print("\nCWRU coverage distances (ep=1 → ep=50):")
for name, vals in cwru_cov.items():
    print(f"  {name}: {vals[0]:.2f} → {vals[-1]:.2f} Hz  "
          f"({vals[0]/vals[-1]:.1f}x improvement)")

print("\nXJTU coverage distances (ep=1 → ep=50):")
for name, vals in xjtu_cov.items():
    print(f"  {name}: {vals[0]:.2f} → {vals[-1]:.2f} Hz  "
          f"({vals[0]/vals[-1]:.1f}x improvement)")

print("\nLoading XJTU KDE at ep=50 …")
xjtu_xs, xjtu_ys = kde_curve(XJTU_DIR, 50, 0, 280)

# ── figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6),
                          gridspec_kw={"width_ratios": [1.3, 1.0], "wspace": 0.40})

# ── panel (a): CWRU vs XJTU coverage distance curves ─────────────────────────
ax = axes[0]
ax.set_title("(a) Nearest-state Coverage Distance vs Epoch")

for name in CWRU_FAULTS:
    c = COLORS[name]
    ax.plot(EPOCHS, cwru_cov[name], color=c, lw=2.0, ls="-",
            marker="o", markersize=5,
            label=f"CWRU {name}")

for name in XJTU_FAULTS:
    c = COLORS[name]
    ax.plot(EPOCHS, xjtu_cov[name], color=c, lw=1.4, ls="--",
            marker="s", markersize=4, markerfacecolor="white", markeredgewidth=1.2,
            label=f"XJTU {name}")

# annotate CWRU improvement at BPFO
bpfo_cwru = cwru_cov["BPFO"]
ratio_c = bpfo_cwru[0] / bpfo_cwru[-1]
ax.annotate(
    f"CWRU BPFO:\n{bpfo_cwru[0]:.1f}→{bpfo_cwru[-1]:.1f} Hz\n({ratio_c:.0f}x)",
    xy=(50, bpfo_cwru[-1]),
    xytext=(28, 12),
    fontsize=7, color=COLORS["BPFO"],
    arrowprops=dict(arrowstyle="->", color=COLORS["BPFO"], lw=0.9),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
)
# annotate XJTU improvement at BPFI
bpfi_xjtu = xjtu_cov["BPFI"]
ratio_x = bpfi_xjtu[0] / bpfi_xjtu[-1]
ax.annotate(
    f"XJTU BPFI:\n{bpfi_xjtu[0]:.1f}→{bpfi_xjtu[-1]:.1f} Hz\n({ratio_x:.0f}x)",
    xy=(50, bpfi_xjtu[-1]),
    xytext=(28, 5),
    fontsize=7, color=COLORS["BPFI"],
    arrowprops=dict(arrowstyle="->", color=COLORS["BPFI"], lw=0.9, ls=":"),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
)

ax.set_xlabel("Training epoch")
ax.set_ylabel("Median coverage distance (Hz)")
ax.set_xlim(-2, 57)
ax.set_xticks(EPOCHS)
ax.set_ylim(bottom=0)
ax.grid(axis="y", ls=":", alpha=0.45)

# separate legends: solid=CWRU, dashed=XJTU
from matplotlib.lines import Line2D
leg_handles = [
    Line2D([0], [0], color="k", lw=2.0, ls="-",  label="CWRU (SNR=0 dB)"),
    Line2D([0], [0], color="k", lw=1.4, ls="--", label="XJTU (Cond3, 2400 rpm)"),
] + [
    Line2D([0], [0], color=COLORS[n], lw=1.5, label=n)
    for n in CWRU_FAULTS
]
ax.legend(handles=leg_handles, loc="upper right", fontsize=7.5, framealpha=0.85,
          ncol=1)

# ── panel (b): XJTU f_bar KDE at ep=50 ──────────────────────────────────────
ax2 = axes[1]
ax2.set_title("(b) XJTU Instantaneous Freq. (ep=50)")

if xjtu_xs is not None:
    ax2.fill_between(xjtu_xs, xjtu_ys / xjtu_ys.max(),
                     alpha=0.35, color="#4393c3")
    ax2.plot(xjtu_xs, xjtu_ys / xjtu_ys.max(),
             color="#4393c3", lw=1.6)

for name, freq in XJTU_FAULTS.items():
    c = COLORS[name]
    ax2.axvline(freq, color=c, lw=0.9, ls="--")
    ax2.text(freq + 1.5, 0.95, name, fontsize=7, color=c, va="top", rotation=90)

ax2.set_xlabel("Instantaneous frequency (Hz)")
ax2.set_ylabel("Normalised density")
ax2.set_xlim(0, 280)
ax2.set_ylim(0, 1.08)
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(20))
ax2.grid(axis="x", ls=":", alpha=0.25)

# ── finalize ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "BearMamba-3 / L_kin mechanism confirmed on both datasets — state frequencies converge to fault targets\n"
    "XJTU accuracy insensitive to alignment (run-to-failure: amplitude-driven classification, not frequency-driven)",
    fontsize=7.5, color="#444444"
)
fig.tight_layout(rect=[0, 0, 1, 0.90])

out_stem = "i1_coverage_cwru_vs_xjtu"
fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=200)
print(f"\nSaved → {OUT_DIR / out_stem}.{{pdf,png}}")
plt.close(fig)
