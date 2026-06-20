#!/usr/bin/env python3
"""
Step 5 / V3: SNR curve — paradigm comparison figure.

Two panels:
  (a) BN-dependent paradigm: SK-SVM / Transformer-1D / 1D-CNN (BN)
  (b) Physical inductive bias: 1D-CNN (BN) / 1D-CNN (no BN) / BM3 CE / BM3+Lkin

Single-panel figure also saved (all baselines).

Usage:
    source ~/论文8/venv/bin/activate
    python experiments/step5_snr_curve_paradigm.py
"""
import json, glob, pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT    = pathlib.Path("/home/jeffwork/论文8/results")
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SNRS = [-8, -6, -4, -2, 0, 10]   # EAAI evaluation grid; 10 = "clean"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_seeds(dirpath):
    """Load seed_*.json files; return (mean%, std%), ddof=1."""
    files = sorted(glob.glob(str(ROOT / dirpath / "seed_*.json")))
    if not files:
        return np.nan, np.nan
    accs = []
    for f in files:
        d = json.loads(pathlib.Path(f).read_text())
        v = d.get("best_val_acc")
        if v is not None:
            accs.append(v * 100.0 if v < 2.0 else v)
    if not accs:
        return np.nan, np.nan
    return float(np.mean(accs)), float(np.std(accs, ddof=1))


def load_svm_metrics(dirpath):
    """SK-SVM stores mean/std in metrics.json with mean_acc/std_acc keys."""
    f = ROOT / dirpath / "metrics.json"
    if not f.exists():
        return np.nan, np.nan
    d = json.loads(f.read_text())
    m = d.get("mean_acc", np.nan)
    s = d.get("std_acc", np.nan)
    # stored as fractions 0–1; convert if needed
    if m < 2.0:
        m, s = m * 100.0, s * 100.0
    return float(m), float(s)


def series(loaders):
    """loaders: list of (snr_value, loader_fn) pairs."""
    means, stds = [], []
    for snr, fn in loaders:
        m, s = fn
        means.append(m)
        stds.append(0.0 if np.isnan(s) else s)
    return np.array(means, float), np.array(stds, float)


# ── build loaders ─────────────────────────────────────────────────────────────

def snr_tag_02(snr):
    """exp02 uses snr-8, snr0, snr10 (no leading +)."""
    return f"snr{snr}"   # works: -8 → snr-8, 0 → snr0, 10 → snr10

def snr_tag_07_clean(snr):
    """exp07 and exp_mext_e21b: negative=snr-N, zero=snr0, +10=clean."""
    if snr == 10:
        return "clean"
    return f"snr{snr}"   # snr-8, snr0


def build_data():
    data = {}

    # ── SK-SVM ──────────────────────────────────────────────────────────────
    loaders = []
    for snr in SNRS:
        tag = "clean" if snr == 10 else f"snr{snr}"
        loaders.append((snr, load_svm_metrics(f"exp07_sksvm_{tag}")))
    data["SK-SVM"] = series(loaders)

    # ── 1D-CNN (BN) ─────────────────────────────────────────────────────────
    loaders = []
    for snr in SNRS:
        tag = snr_tag_07_clean(snr)
        loaders.append((snr, load_seeds(f"exp07_cnn1d_{tag}")))
    data["1D-CNN (BN)"] = series(loaders)

    # ── Transformer-1D ──────────────────────────────────────────────────────
    loaders = []
    for snr in SNRS:
        tag = snr_tag_07_clean(snr)
        loaders.append((snr, load_seeds(f"exp07_transformer1d_{tag}")))
    data["Transformer-1D"] = series(loaders)

    # ── BearMamba-2 CE-only ──────────────────────────────────────────────────
    loaders = []
    for snr in SNRS:
        tag = snr_tag_02(snr)
        loaders.append((snr, load_seeds(f"exp04_mamba2_{tag}")))
    data["BearMamba-2"] = series(loaders)

    # ── BM3 CE-only ─────────────────────────────────────────────────────────
    # SNR in [-8..0]: exp02_snrX_nokin; clean (+10): exp01_cwru_baseline
    loaders = []
    for snr in SNRS:
        if snr == 10:
            loaders.append((snr, load_seeds("exp01_cwru_baseline")))
        else:
            loaders.append((snr, load_seeds(f"exp02_snr{snr}_nokin")))
    data["BM3 CE-only"] = series(loaders)

    # ── BM3 +L_kin ──────────────────────────────────────────────────────────
    loaders = []
    for snr in SNRS:
        if snr == 10:
            loaders.append((snr, load_seeds("exp01_cwru_kin")))
        else:
            loaders.append((snr, load_seeds(f"exp02_snr{snr}_kin")))
    data["BM3 + $\\mathcal{L}_{\\mathrm{kin}}$"] = series(loaders)

    # ── 1D-CNN no-BN (Phase 2b) ─────────────────────────────────────────────
    loaders = []
    for snr in SNRS:
        tag = snr_tag_07_clean(snr)
        loaders.append((snr, load_seeds(f"exp_mext_e21b_1dcnn_nobn_cwru_{tag}")))
    data["1D-CNN (no BN)"] = series(loaders)

    return data


data = build_data()

# ── sanity check ─────────────────────────────────────────────────────────────
print("=== Data verification ===")
for name, (m, s) in data.items():
    row = "  ".join(f"{v:.1f}" if not np.isnan(v) else "NaN" for v in m)
    print(f"  {name:45s} {row}")

# ── style ─────────────────────────────────────────────────────────────────────
COLORS = {
    "SK-SVM":                                    "#888888",
    "1D-CNN (BN)":                               "#d62728",
    "1D-CNN (no BN)":                            "#ff7f0e",
    "Transformer-1D":                            "#9467bd",
    "BearMamba-2":                               "#1f77b4",
    "BM3 CE-only":                               "#2ca02c",
    "BM3 + $\\mathcal{L}_{\\mathrm{kin}}$":     "#17becf",
}
LS = {
    "SK-SVM":                                    (0, (3, 1, 1, 1)),
    "1D-CNN (BN)":                               "-",
    "1D-CNN (no BN)":                            "--",
    "Transformer-1D":                            "-.",
    "BearMamba-2":                               "-",
    "BM3 CE-only":                               "--",
    "BM3 + $\\mathcal{L}_{\\mathrm{kin}}$":     "-",
}

x = np.array(SNRS)


def plot_methods(ax, names):
    for name in names:
        m, s = data[name]
        ok = ~np.isnan(m)
        if not ok.any():
            continue
        ax.plot(x[ok], m[ok], color=COLORS[name], ls=LS[name],
                marker="o", ms=4, lw=1.5, label=name)
        ax.fill_between(x[ok], m[ok] - s[ok], m[ok] + s[ok],
                        alpha=0.12, color=COLORS[name])


# ── Figure 1: two-panel paradigm ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)

ax = axes[0]
ax.set_title("(a) BN-dependent paradigm", loc="left", fontsize=8.5)
plot_methods(ax, ["SK-SVM", "Transformer-1D", "1D-CNN (BN)"])
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Accuracy (%)")
ax.set_xticks(SNRS); ax.set_ylim(55, 102)
ax.legend(loc="lower right", framealpha=0.85)
ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5)

ax = axes[1]
ax.set_title("(b) Physical inductive bias vs. BN ablation", loc="left", fontsize=8.5)
plot_methods(ax, ["1D-CNN (BN)", "1D-CNN (no BN)",
                  "BM3 CE-only", "BM3 + $\\mathcal{L}_{\\mathrm{kin}}$"])
ax.set_xlabel("SNR (dB)")
ax.set_xticks(SNRS); ax.set_ylim(55, 102)
ax.legend(loc="lower right", framealpha=0.85)
ax.grid(axis="y", ls=":", lw=0.5, alpha=0.5)

# annotate BN gap at −8 dB
m_bn, _  = data["1D-CNN (BN)"]
m_nobn,_ = data["1D-CNN (no BN)"]
if not (np.isnan(m_bn[0]) or np.isnan(m_nobn[0])):
    gap = m_bn[0] - m_nobn[0]
    mid = (m_bn[0] + m_nobn[0]) / 2
    ax.annotate(f"BN removed:\n$-{gap:.1f}$ pp",
                xy=(-8, mid), xytext=(-6.2, mid - 5.5), fontsize=6.5,
                color=COLORS["1D-CNN (BN)"],
                arrowprops=dict(arrowstyle="->", color=COLORS["1D-CNN (BN)"],
                                lw=0.8, shrinkA=2, shrinkB=2))

fig.suptitle(r"CWRU 4-class accuracy vs.\ SNR  ($\pm1\sigma$, 5 seeds, EAAI grid)",
             fontsize=9)
fig.tight_layout()
for fmt in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"snr_curve_paradigm.{fmt}",
                bbox_inches="tight", dpi=200 if fmt == "png" else None)
print(f"\nSaved  results/figures/snr_curve_paradigm.{{pdf,png}}")

# ── Figure 2: all-baselines single panel ─────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(5.5, 3.6))
ORDER = ["SK-SVM", "Transformer-1D", "1D-CNN (BN)", "1D-CNN (no BN)",
         "BearMamba-2", "BM3 CE-only", "BM3 + $\\mathcal{L}_{\\mathrm{kin}}$"]
plot_methods(ax2, ORDER)
ax2.set_xlabel("SNR (dB)"); ax2.set_ylabel("Accuracy (%)")
ax2.set_xticks(SNRS); ax2.set_ylim(55, 102)
ax2.legend(loc="lower right", fontsize=7, framealpha=0.9)
ax2.grid(axis="y", ls=":", lw=0.5, alpha=0.5)
ax2.set_title("CWRU 4-class: all baselines  ($\\pm1\\sigma$, 5 seeds)", fontsize=9)
fig2.tight_layout()
for fmt in ("pdf", "png"):
    fig2.savefig(OUT_DIR / f"snr_curve_allbaselines.{fmt}",
                 bbox_inches="tight", dpi=200 if fmt == "png" else None)
print(f"Saved  results/figures/snr_curve_allbaselines.{{pdf,png}}")
plt.close("all")
print("Done.")
