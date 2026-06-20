#!/usr/bin/env python3
"""
Step 5 / B2: Single vs dual-sensor SNR curve (C3 multi-sensor evidence).

2-panel figure:
  (a) Accuracy vs SNR — single vs dual × ±L_kin  (shaded ±1std bands)
  (b) Gain (dual − single) vs SNR  (annotate -4dB nokin p=0.031)

Key finding: dual-sensor consistently outperforms single across all SNR;
strongest at low SNR (-8dB Δ≈+2.5pp); -4dB nokin Wilcoxon p=0.031 (唯一显著).

SNRS: {-8,-6,-4,-2,0}dB  (+10dB excluded: single already at 100.00%)
"""
import json
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats

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

SNRS = [-8, -6, -4, -2, 0]   # +10dB excluded


def load(exp_dir: pathlib.Path):
    p = exp_dir / "summary.json"
    if not p.exists():
        return None, None, None
    d = json.loads(p.read_text())
    seeds = np.array(d.get("best_val_accs", [])) * 100
    if len(seeds) == 0:
        return None, None, None
    return seeds, float(seeds.mean()), float(seeds.std(ddof=1))


def get_snr_dir(base: str, snr: int) -> pathlib.Path:
    if snr == 0:
        return ROOT / base
    return ROOT / f"{base}_snr{'m' if snr < 0 else ''}{abs(snr)}"


# ── load data ─────────────────────────────────────────────────────────────────
data = {}
for cond, single_base, dual_base in [
    ("nokin", "exp02_snr{snr}_nokin", "exp_b2_dual_nokin"),
    ("kin",   "exp02_snr{snr}_kin",   "exp_b2_dual_kin"),
]:
    data[cond] = {"single": {"seeds": [], "mean": [], "std": []},
                  "dual":   {"seeds": [], "mean": [], "std": []}}
    for snr in SNRS:
        single_dir = ROOT / single_base.format(snr=snr)
        dual_dir   = get_snr_dir(dual_base, snr)

        s_seeds, s_m, s_s = load(single_dir)
        d_seeds, d_m, d_s = load(dual_dir)

        data[cond]["single"]["seeds"].append(s_seeds)
        data[cond]["single"]["mean"].append(s_m)
        data[cond]["single"]["std"].append(s_s)
        data[cond]["dual"]["seeds"].append(d_seeds)
        data[cond]["dual"]["mean"].append(d_m)
        data[cond]["dual"]["std"].append(d_s)

# Wilcoxon p-values (dual vs single, per SNR)
p_nokin, p_kin = [], []
for i, snr in enumerate(SNRS):
    sn = data["nokin"]["single"]["seeds"][i]
    dn = data["nokin"]["dual"]["seeds"][i]
    sk = data["kin"]["single"]["seeds"][i]
    dk = data["kin"]["dual"]["seeds"][i]
    try:
        # one-sided: H1: dual > single (consistent with analyze_b2.py)
        pn = stats.wilcoxon(dn, sn, alternative="greater").pvalue; p_nokin.append(pn)
        pk = stats.wilcoxon(dk, sk, alternative="greater").pvalue; p_kin.append(pk)
    except Exception:
        p_nokin.append(float("nan")); p_kin.append(float("nan"))

print("B2 Wilcoxon p-values (dual vs single, n=5):")
for snr, pn, pk in zip(SNRS, p_nokin, p_kin):
    sig_n = "*" if pn < 0.05 else " "
    sig_k = "*" if pk < 0.05 else " "
    delta_n = (np.mean(data["nokin"]["dual"]["seeds"][SNRS.index(snr)]) -
               np.mean(data["nokin"]["single"]["seeds"][SNRS.index(snr)])) if data["nokin"]["dual"]["seeds"][SNRS.index(snr)] is not None else np.nan
    print(f"  {snr:+3d}dB  nokin p={pn:.4f}{sig_n}  kin p={pk:.4f}{sig_k}  Δnokin≈{delta_n:+.2f}pp")

snrs = np.array(SNRS, dtype=float)

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6),
                          gridspec_kw={"width_ratios": [2.0, 1.0], "wspace": 0.38})

COLOR = {"nokin": "#1f77b4", "kin": "#2ca02c"}
LABEL = {"nokin": "CE-only", "kin": "+L_kin"}

# ── panel (a): accuracy curves ────────────────────────────────────────────────
ax = axes[0]
ax.set_title("(a) Single vs Dual Sensor — Accuracy vs SNR")

for cond in ("nokin", "kin"):
    c = COLOR[cond]
    lbl = LABEL[cond]
    m_s = np.array(data[cond]["single"]["mean"])
    s_s = np.array(data[cond]["single"]["std"])
    m_d = np.array(data[cond]["dual"]["mean"])
    s_d = np.array(data[cond]["dual"]["std"])

    # single: dashed
    ax.fill_between(snrs, m_s - s_s, m_s + s_s, alpha=0.18, color=c)
    ax.plot(snrs, m_s, color=c, lw=1.4, ls="--",
            marker="o", markersize=5, markerfacecolor="white", markeredgewidth=1.2,
            label=f"Single {lbl}")
    # dual: solid
    ax.fill_between(snrs, m_d - s_d, m_d + s_d, alpha=0.30, color=c)
    ax.plot(snrs, m_d, color=c, lw=2.0, ls="-",
            marker="o", markersize=5,
            label=f"Dual {lbl}")

ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Accuracy (%)")
ax.set_xlim(-9.5, 1.5)
ax.set_xticks(SNRS)
ax.set_xticklabels([f"{s:+d}" for s in SNRS])
ax.set_ylim(82, 101.5)
ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.grid(axis="y", ls=":", alpha=0.45)
ax.grid(axis="x", ls=":", alpha=0.25)
ax.legend(loc="lower right", framealpha=0.88, ncol=2)

# mark p=0.031 at -4dB nokin
idx_m4 = SNRS.index(-4)
y_dual_m4 = data["nokin"]["dual"]["mean"][idx_m4]
ax.annotate(
    "p = 0.031\n(n=5, Wilcoxon)",
    xy=(-4, y_dual_m4 + 0.2),
    xytext=(-5.5, 99.5),
    fontsize=7.5, color=COLOR["nokin"],
    arrowprops=dict(arrowstyle="->", color=COLOR["nokin"], lw=0.9),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.88, ec=COLOR["nokin"]),
)

# ── panel (b): gain (dual - single) ──────────────────────────────────────────
ax2 = axes[1]
ax2.set_title("(b) Gain: Dual − Single (pp)")

for cond in ("nokin", "kin"):
    c = COLOR[cond]
    lbl = LABEL[cond]
    gain = np.array(data[cond]["dual"]["mean"]) - np.array(data[cond]["single"]["mean"])
    ax2.plot(snrs, gain, color=c, lw=1.8, ls="-" if cond == "nokin" else "--",
             marker="o", markersize=5,
             label=f"{lbl}")
    for i, (snr, g) in enumerate(zip(SNRS, gain)):
        if abs(g) > 0.3:
            ax2.text(snr + 0.15, g + 0.05, f"+{g:.2f}", fontsize=6.5,
                     color=c, va="bottom")

ax2.axhline(0, color="#aaaaaa", lw=0.8, ls=":")
ax2.set_xlabel("SNR (dB)")
ax2.set_ylabel("Accuracy gain (pp)")
ax2.set_xlim(-9.5, 1.5)
ax2.set_xticks(SNRS)
ax2.set_xticklabels([f"{s:+d}" for s in SNRS])
ax2.set_ylim(bottom=-0.1)
ax2.grid(axis="y", ls=":", alpha=0.45)
ax2.legend(loc="upper right", framealpha=0.88)

# mark p=0.031 at -4dB
gain_m4_nokin = data["nokin"]["dual"]["mean"][idx_m4] - data["nokin"]["single"]["mean"][idx_m4]
ax2.annotate("p=0.031 *", xy=(-4, gain_m4_nokin),
             xytext=(-6, gain_m4_nokin + 0.4),
             fontsize=7.5, color=COLOR["nokin"],
             arrowprops=dict(arrowstyle="->", color=COLOR["nokin"], lw=0.9))

# ── finalize ─────────────────────────────────────────────────────────────────
fig.suptitle(
    "CWRU SISO→Dual (DE+FE) / BearMamba-3 / 5 seeds — C3 Multi-sensor Evidence",
    fontsize=8, color="#444444"
)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out_stem = "b2_snr_curve_dual"
fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=200)
print(f"\nSaved → {OUT_DIR / out_stem}.{{pdf,png}}")
plt.close(fig)
