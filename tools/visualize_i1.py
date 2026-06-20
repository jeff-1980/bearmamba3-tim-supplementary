"""
I1 Kinematic Frequency Snapshot Visualisation — Step 5
Generates two publication-quality figures:
  1. i1_freq_distribution_cwru.pdf  — KDE + coverage convergence (CWRU SNR=0 kin)
  2. i1_coverage_cwru_vs_xjtu.pdf   — CWRU vs XJTU coverage curves + XJTU KDE

Run from ~/论文8/paper/:
    python3 ../tools/visualize_i1.py
Output: ../results/figures/i1_freq_distribution_cwru.pdf
        ../results/figures/i1_coverage_cwru_vs_xjtu.pdf
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import glob, os, sys

# ── paths ──────────────────────────────────────────────────────────────────────
RESULTS = os.path.expanduser('~/论文8/results')
FIG_DIR = os.path.join(RESULTS, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── bearing fault frequencies ──────────────────────────────────────────────────
CWRU_FREQS  = {'BPFO': 107.36, r'$2\times$BSF': 141.17, 'BPFI': 162.19}
XJTU_FREQS  = {'BPFO': 123.32, r'$2\times$BSF': 165.33, 'BPFI': 196.68}

FREQ_COLORS = {'BPFO': '#e41a1c', r'$2\times$BSF': '#ff7f00', 'BPFI': '#377eb8'}
EPOCHS      = [1, 10, 20, 30, 40, 50]

# ── matplotlib style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'pdf.fonttype': 42,   # TrueType in PDF — Elsevier requirement
    'ps.fonttype':  42,
})

# ── helpers ────────────────────────────────────────────────────────────────────

def load_epoch(exp_dir, epoch):
    """Return concatenated f_bar (N,256) for all seeds at a given epoch."""
    pat = os.path.join(RESULTS, exp_dir, f'*ep{epoch:03d}.npz')
    files = sorted(glob.glob(pat))
    if not files:
        return None
    return np.concatenate([np.load(f)['f_bar'] for f in files], axis=0)


def coverage_curve(exp_dir, fault_freqs_dict):
    """
    For each epoch, compute mean nearest-state coverage distance (Hz) per fault frequency.
    Returns dict: {name: np.array shape (len(EPOCHS),)}
    """
    result = {k: [] for k in fault_freqs_dict}
    for ep in EPOCHS:
        f_bar = load_epoch(exp_dir, ep)
        if f_bar is None:
            for k in result: result[k].append(np.nan)
            continue
        for name, fc in fault_freqs_dict.items():
            dists = np.min(np.abs(f_bar - fc), axis=1)
            result[name].append(dists.mean())
    return {k: np.array(v) for k, v in result.items()}


def zone_kde(f_bar, lo, hi, n_pts=300):
    """KDE of f_bar values within [lo, hi] Hz."""
    vals = f_bar[(f_bar >= lo) & (f_bar <= hi)].ravel()
    if len(vals) < 10:
        return np.linspace(lo, hi, n_pts), np.zeros(n_pts)
    x = np.linspace(lo, hi, n_pts)
    kde = gaussian_kde(vals, bw_method='silverman')
    return x, kde(x)

# ── epoch colormap: light gray → dark blue ────────────────────────────────────
EPOCH_CMAP = plt.cm.Blues
ep_norm    = mcolors.Normalize(vmin=0, vmax=len(EPOCHS) + 1)

def ep_color(idx):
    return EPOCH_CMAP(ep_norm(idx + 1))   # idx=0 → light, idx=5 → dark

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — i1_freq_distribution_cwru.pdf
# ══════════════════════════════════════════════════════════════════════════════
print("Loading CWRU SNR=0 kin snapshots …")

EXP_CWRU = 'exp02_snr0_kin'
ZONE_LO, ZONE_HI = 60.0, 230.0   # Hz — fault frequency zone

fig1, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
ax_kde_full, ax_kde_zone, ax_cov = axes

# --- (a) Full-range KDE (0 – 300 Hz), highlighting fault zone ----------------
ax = ax_kde_full
for idx, ep in enumerate(EPOCHS):
    f_bar = load_epoch(EXP_CWRU, ep)
    if f_bar is None: continue
    vals = f_bar.ravel()
    vals = vals[(vals > 0) & (vals < 300)]
    x = np.linspace(0, 300, 500)
    kde = gaussian_kde(vals, bw_method='silverman')
    y = kde(x)
    alpha = 0.4 + 0.12 * idx
    ax.plot(x, y, color=ep_color(idx), lw=1.2, alpha=alpha)

ax.axvspan(ZONE_LO, ZONE_HI, color='#ffeb3b', alpha=0.18, label='Fault-freq zone')
for name, fc in CWRU_FREQS.items():
    ax.axvline(fc, color=FREQ_COLORS[name], ls='--', lw=0.9, alpha=0.8)

ax.set_xlabel('State frequency (Hz)')
ax.set_ylabel('KDE density')
ax.set_xlim(0, 300)
ax.set_title('(a) All frequencies, epoch 1→50')
ax.tick_params(axis='both', which='both', length=3)

# ── legend: epoch colour scale ─────────────────────────────────────────────
ep_handles = [
    Line2D([0], [0], color=ep_color(0), lw=1.2, label='Epoch 1'),
    Line2D([0], [0], color=ep_color(2), lw=1.2, label='Epoch 20'),
    Line2D([0], [0], color=ep_color(5), lw=1.2, label='Epoch 50'),
]
ax.legend(handles=ep_handles, loc='upper right', framealpha=0.8)

# --- (b) Zoomed KDE in fault-frequency zone ----------------------------------
ax = ax_kde_zone
for idx, ep in enumerate(EPOCHS):
    f_bar = load_epoch(EXP_CWRU, ep)
    if f_bar is None: continue
    x, y = zone_kde(f_bar, ZONE_LO, ZONE_HI)
    alpha = 0.4 + 0.12 * idx
    ax.plot(x, y, color=ep_color(idx), lw=1.3, alpha=alpha)

for name, fc in CWRU_FREQS.items():
    ax.axvline(fc, color=FREQ_COLORS[name], ls='--', lw=1.1, alpha=0.9, label=name)

ax.set_xlabel('State frequency (Hz)')
ax.set_ylabel('')
ax.set_title('(b) Fault-frequency zone (60–230 Hz)')
ax.set_xlim(ZONE_LO, ZONE_HI)
ax.legend(loc='upper left', framealpha=0.8, fontsize=7.5)
ax.tick_params(axis='both', which='both', length=3)

# annotate BPFO, 2BSF, BPFI values
for name, fc in CWRU_FREQS.items():
    ax.text(fc + 1.5, ax.get_ylim()[1] * 0.95, f'{fc:.0f}',
            color=FREQ_COLORS[name], fontsize=6.5, va='top', ha='left')

# --- (c) Coverage-distance curve vs epoch ------------------------------------
ax = ax_cov
print("  Computing CWRU coverage curves …")
cov_cwru = coverage_curve(EXP_CWRU, CWRU_FREQS)

for name, vals in cov_cwru.items():
    ax.plot(EPOCHS, vals, marker='o', markersize=4,
            color=FREQ_COLORS[name], label=name)
    ax.annotate(
        f'{vals[0]:.1f} Hz',
        (EPOCHS[0], vals[0]), textcoords='offset points', xytext=(4, 2),
        fontsize=6.5, color=FREQ_COLORS[name]
    )
    ax.annotate(
        f'{vals[-1]:.1f} Hz',
        (EPOCHS[-1], vals[-1]), textcoords='offset points', xytext=(-30, -8),
        fontsize=6.5, color=FREQ_COLORS[name]
    )

ax.set_xlabel('Training epoch')
ax.set_ylabel('Coverage distance (Hz)')
ax.set_title('(c) Nearest-state distance to fault harmonics')
ax.set_xlim(0, 55)
ax.set_ylim(bottom=0)
ax.legend(loc='upper right', framealpha=0.8)
ax.tick_params(axis='both', which='both', length=3)

fig1.tight_layout(pad=1.0)
out1 = os.path.join(FIG_DIR, 'i1_freq_distribution_cwru.pdf')
fig1.savefig(out1, bbox_inches='tight')
print(f"  → {out1}")
plt.close(fig1)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — i1_coverage_cwru_vs_xjtu.pdf
# ══════════════════════════════════════════════════════════════════════════════
print("Loading XJTU cross-condition kin snapshots …")

EXP_XJTU = 'exp_xjtu_cross_kin'

fig2, (ax_cov2, ax_kde2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

# --- (a) Coverage curves: CWRU (solid) vs XJTU (dashed) ----------------------
print("  Computing XJTU coverage curves …")
cov_xjtu = coverage_curve(EXP_XJTU, XJTU_FREQS)

# Use same label names but different actual frequencies; colour by name position
LABEL_LIST = ['BPFO', r'$2\times$BSF', 'BPFI']
for name in LABEL_LIST:
    col = FREQ_COLORS[name]
    # CWRU
    if name in cov_cwru:
        ax_cov2.plot(EPOCHS, cov_cwru[name], color=col, ls='-',
                     marker='o', markersize=4, lw=1.4, label=f'CWRU {name}')
    # XJTU
    if name in cov_xjtu:
        ax_cov2.plot(EPOCHS, cov_xjtu[name], color=col, ls='--',
                     marker='s', markersize=4, lw=1.4, label=f'XJTU {name}')

ax_cov2.set_xlabel('Training epoch')
ax_cov2.set_ylabel('Mean nearest-state distance (Hz)')
ax_cov2.set_title('(a) Coverage-distance convergence')
ax_cov2.set_xlim(0, 55)
ax_cov2.set_ylim(bottom=0)
ax_cov2.tick_params(axis='both', which='both', length=3)

# custom legend: solid=CWRU, dashed=XJTU; then colour legend
leg_style = [
    Line2D([0], [0], color='k', ls='-',  lw=1.4, marker='o', ms=4, label='CWRU'),
    Line2D([0], [0], color='k', ls='--', lw=1.4, marker='s', ms=4, label='XJTU'),
]
leg_color = [
    Line2D([0], [0], color=FREQ_COLORS[n], lw=1.8, label=n)
    for n in LABEL_LIST
]
ax_cov2.legend(handles=leg_style + leg_color, ncol=2,
               loc='upper right', framealpha=0.85, fontsize=7.5)

# annotate XJTU BPFI convergence ratio
bpfi_xjtu = cov_xjtu.get('BPFI', None)
if bpfi_xjtu is not None:
    ratio = bpfi_xjtu[0] / bpfi_xjtu[-1]
    ax_cov2.annotate(
        f'XJTU BPFI: {bpfi_xjtu[0]:.1f}→{bpfi_xjtu[-1]:.1f} Hz\n({ratio:.1f}× reduction)',
        xy=(EPOCHS[-1], bpfi_xjtu[-1]),
        xytext=(35, 10), textcoords='offset points',
        fontsize=6.5, color=FREQ_COLORS['BPFI'],
        arrowprops=dict(arrowstyle='->', color=FREQ_COLORS['BPFI'], lw=0.8)
    )

# --- (b) XJTU state KDE at epoch 50, with fault-frequency lines --------------
ax = ax_kde2
f_bar_xjtu_ep50 = load_epoch(EXP_XJTU, 50)

# Get the XJTU fs_eff from one file
xjtu_ep50_files = sorted(glob.glob(os.path.join(RESULTS, EXP_XJTU, '*ep050.npz')))
fs_eff_xjtu = float(np.load(xjtu_ep50_files[0])['fs_eff']) if xjtu_ep50_files else 12800.0

XJTU_ZONE_LO, XJTU_ZONE_HI = 80.0, 280.0

if f_bar_xjtu_ep50 is not None:
    x_xjtu, y_xjtu = zone_kde(f_bar_xjtu_ep50, XJTU_ZONE_LO, XJTU_ZONE_HI)
    ax.plot(x_xjtu, y_xjtu, color='#1a9641', lw=1.5, label='State freq KDE\n(XJTU ep50)')
    ax.fill_between(x_xjtu, y_xjtu, alpha=0.15, color='#1a9641')

for name, fc in XJTU_FREQS.items():
    ax.axvline(fc, color=FREQ_COLORS[name], ls='--', lw=1.1, alpha=0.9, label=name)
    ax.text(fc + 1.5, ax.get_ylim()[1] * 0.95, f'{fc:.0f}',
            color=FREQ_COLORS[name], fontsize=6.5, va='top', ha='left')

ax.set_xlabel('State frequency (Hz)')
ax.set_ylabel('KDE density')
ax.set_title('(b) XJTU state freq KDE at epoch 50\n(Cond3, 2400 rpm)')
ax.set_xlim(XJTU_ZONE_LO, XJTU_ZONE_HI)
ax.legend(loc='upper left', framealpha=0.85, fontsize=7.5)
ax.tick_params(axis='both', which='both', length=3)

fig2.tight_layout(pad=1.0)
out2 = os.path.join(FIG_DIR, 'i1_coverage_cwru_vs_xjtu.pdf')
fig2.savefig(out2, bbox_inches='tight')
print(f"  → {out2}")
plt.close(fig2)

print("\nDone. Both figures written to results/figures/")
