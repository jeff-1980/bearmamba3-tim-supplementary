#!/usr/bin/env python3
"""
Step 5 / SNR curve: accuracy vs SNR for BM3 CE-only / BM3 +L_kin / BM2 CE-only.

Key C2 visual: BM3+L_kin has noticeably narrower std bands at low SNR,
especially at -4dB (std: 1.01% → 0.38%, -62%).

Data: results/exp02_snr*/summary.json (BM3) and results/exp04_mamba2_snr*/summary.json (BM2)
EAAI grid: {-8, -6, -4, -2, 0, +10} dB

Usage:
    source venv/bin/activate
    python experiments/step5_snr_curve.py
"""
import json
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

ROOT    = pathlib.Path("/home/jeffwork/论文8/results")
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SNRS = [-8, -6, -4, -2, 0, 10]   # EAAI grid


def load_accs(exp_dir: pathlib.Path):
    """Return (mean%, std%) from summary.json, or (None, None) if missing."""
    p = exp_dir / "summary.json"
    if not p.exists():
        return None, None
    d = json.loads(p.read_text())
    seeds = d.get("best_val_accs", [])
    if not seeds:
        return None, None
    arr = np.array(seeds) * 100
    return float(arr.mean()), float(arr.std(ddof=1))


# ── load data ─────────────────────────────────────────────────────────────────
bm3_nokin_m, bm3_nokin_s = [], []
bm3_kin_m,   bm3_kin_s   = [], []
bm2_m,       bm2_s       = [], []

for snr in SNRS:
    m, s = load_accs(ROOT / f"exp02_snr{snr}_nokin");  bm3_nokin_m.append(m); bm3_nokin_s.append(s)
    m, s = load_accs(ROOT / f"exp02_snr{snr}_kin");    bm3_kin_m.append(m);   bm3_kin_s.append(s)
    m, s = load_accs(ROOT / f"exp04_mamba2_snr{snr}"); bm2_m.append(m);       bm2_s.append(s)

snrs    = np.array(SNRS, dtype=float)
x_ticks = snrs.copy()

# Convert to numpy, handle None
def arr(lst):
    return np.array([v if v is not None else np.nan for v in lst])

bm3_nokin_m = arr(bm3_nokin_m); bm3_nokin_s = arr(bm3_nokin_s)
bm3_kin_m   = arr(bm3_kin_m);   bm3_kin_s   = arr(bm3_kin_s)
bm2_m       = arr(bm2_m);       bm2_s       = arr(bm2_s)

print("SNR data loaded:")
for i, snr in enumerate(SNRS):
    print(f"  SNR={snr:+3d}: nokin={bm3_nokin_m[i]:.2f}±{bm3_nokin_s[i]:.2f}  "
          f"kin={bm3_kin_m[i]:.2f}±{bm3_kin_s[i]:.2f}  "
          f"bm2={bm2_m[i]:.2f}±{bm2_s[i]:.2f}")

# ── figure: 2-panel ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6),
                          gridspec_kw={"width_ratios": [2.2, 1.0], "wspace": 0.35})

# ── panel (a): accuracy curves with std bands ─────────────────────────────────
ax = axes[0]
ax.set_title("(a) Accuracy vs. SNR — BM3 ±L_kin and BM2 baseline")

# BM2 reference (orange, thinner, semi-transparent)
ax.fill_between(snrs, bm2_m - bm2_s, bm2_m + bm2_s,
                alpha=0.18, color="#ff7f0e")
ax.plot(snrs, bm2_m, color="#ff7f0e", lw=1.4, ls="--",
        marker="s", markersize=5, markerfacecolor="white", markeredgewidth=1.2,
        label="BM2 CE-only (baseline)")

# BM3 nokin (blue dashed)
ax.fill_between(snrs, bm3_nokin_m - bm3_nokin_s, bm3_nokin_m + bm3_nokin_s,
                alpha=0.20, color="#1f77b4")
ax.plot(snrs, bm3_nokin_m, color="#1f77b4", lw=1.5, ls="--",
        marker="o", markersize=5, markerfacecolor="white", markeredgewidth=1.2,
        label="BM3 CE-only (w/o L_kin)")

# BM3 kin (blue solid, narrower band)
ax.fill_between(snrs, bm3_kin_m - bm3_kin_s, bm3_kin_m + bm3_kin_s,
                alpha=0.35, color="#1f77b4")
ax.plot(snrs, bm3_kin_m, color="#1f77b4", lw=2.0, ls="-",
        marker="o", markersize=5,
        label="BM3 +L_kin (λ=0.01, cover)")

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Accuracy (%)")
ax.set_xlim(-9.5, 11.5)
ax.set_xticks(SNRS)
ax.set_xticklabels([f"{s:+d}" for s in SNRS])
ax.set_ylim(82, 101.5)
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.grid(axis="y", ls=":", alpha=0.45)
ax.grid(axis="x", ls=":", alpha=0.25)
ax.legend(loc="lower right", framealpha=0.88)

# annotate std reduction at -4dB
snr_m4_idx = SNRS.index(-4)
y_nokin = bm3_nokin_m[snr_m4_idx]
x_m4    = -4
pct = (1 - bm3_kin_s[snr_m4_idx] / bm3_nokin_s[snr_m4_idx]) * 100
ax.annotate(
    f"σ: {bm3_nokin_s[snr_m4_idx]:.2f}% → {bm3_kin_s[snr_m4_idx]:.2f}%  (−{pct:.0f}%)",
    xy=(x_m4, y_nokin - bm3_nokin_s[snr_m4_idx] * 0.6),
    xytext=(-7.5, 94.5),
    fontsize=7.5, color="#1f77b4",
    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.9),
    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85, ec="#aaaaaa"),
)

# ── panel (b): std (variance) curves only ─────────────────────────────────────
ax2 = axes[1]
ax2.set_title("(b) Std across 5 seeds")

ax2.plot(snrs, bm3_nokin_s, color="#1f77b4", lw=1.5, ls="--",
         marker="o", markersize=5, markerfacecolor="white", markeredgewidth=1.2,
         label="BM3 w/o L_kin")
ax2.plot(snrs, bm3_kin_s, color="#1f77b4", lw=2.0, ls="-",
         marker="o", markersize=5,
         label="BM3 +L_kin")
ax2.plot(snrs, bm2_s, color="#ff7f0e", lw=1.4, ls="--",
         marker="s", markersize=5, markerfacecolor="white", markeredgewidth=1.2,
         label="BM2")

ax2.set_xlabel("SNR (dB)")
ax2.set_ylabel("Std (%)")
ax2.set_xlim(-9.5, 11.5)
ax2.set_xticks(SNRS)
ax2.set_xticklabels([f"{s:+d}" for s in SNRS])
ax2.set_ylim(bottom=-0.05)
ax2.grid(axis="y", ls=":", alpha=0.45)
ax2.grid(axis="x", ls=":", alpha=0.25)
ax2.legend(loc="upper right", framealpha=0.88)

# annotate -4dB reduction explicitly
y_bot  = bm3_kin_s[snr_m4_idx] + 0.05
ax2.annotate(
    f"−62%",
    xy=(-4, bm3_kin_s[snr_m4_idx]),
    xytext=(-2.5, bm3_kin_s[snr_m4_idx] + 0.3),
    fontsize=7.5, color="#1f77b4",
    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.9),
)

# ── finalize ─────────────────────────────────────────────────────────────────
fig.suptitle(
    "CWRU 4-class / BearMamba-3 SISO / 5 seeds — EAAI evaluation grid",
    fontsize=8, color="#444444"
)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out_stem = "snr_curve_cwru"
fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=200)
print(f"\nSaved → {OUT_DIR / out_stem}.{{pdf,png}}")
plt.close(fig)
