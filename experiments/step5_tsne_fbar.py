#!/usr/bin/env python3
"""
Step 5 / t-SNE (f_bar): Instantaneous frequency space visualization.

Shows how L_kin training progressively organizes the model's internal
frequency representations (f_bar ∈ R^256) over training epochs.

At ep=1 (before L_kin converges), fault classes are mixed in frequency space.
At ep=50 (converged), DB Index improves by ~18%, indicating more organized
frequency representations — complementary to the I1 coverage-distance figure.

Note: classes are NOT expected to be fully separated in f_bar space because
the cover variant of L_kin pushes ALL states toward ALL fault frequencies
regardless of class. Discrimination is achieved by the downstream classifier.

Data: results/exp02_snr0_kin/ (CWRU SNR=0dB, λ=0.01, 5 seeds)
Output: results/figures/tsne_fbar_cwru_snr0.{pdf,png}

Usage:
    source venv/bin/activate
    python experiments/step5_tsne_fbar.py
"""
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler

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

ROOT    = pathlib.Path("/home/jeffwork/论文8")
SRC_DIR = ROOT / "results" / "exp02_snr0_kin"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES  = {0: "Normal", 1: "Inner", 2: "Ball", 3: "Outer"}
CLASS_COLORS = {0: "#999999", 1: "#2ca02c", 2: "#ff7f0e", 3: "#1f77b4"}
CLASS_MARKERS = {0: "o", 1: "^", 2: "s", 3: "D"}


def load_epoch(epoch: int):
    files = sorted(SRC_DIR.glob(f"*ep{epoch:03d}*.npz"))
    fbar   = np.concatenate([np.load(f)["f_bar"]   for f in files])  # (N, 256)
    labels = np.concatenate([np.load(f)["labels"]  for f in files])  # (N,)
    return fbar, labels


def embed(fbar: np.ndarray, seed: int = 42):
    scaler  = StandardScaler()
    scaled  = scaler.fit_transform(fbar)
    tsne    = TSNE(n_components=2, perplexity=40, max_iter=1000,
                   random_state=seed, init="pca", learning_rate="auto")
    emb_2d  = tsne.fit_transform(scaled)
    return emb_2d, scaled


# ─── load & embed ─────────────────────────────────────────────────────────────
print("Loading epoch 1 …")
fbar1, labels1 = load_epoch(1)
emb1, sc1 = embed(fbar1)
db1 = davies_bouldin_score(sc1, labels1)

print("Loading epoch 50 …")
fbar50, labels50 = load_epoch(50)
emb50, sc50 = embed(fbar50)
db50 = davies_bouldin_score(sc50, labels50)

print(f"DB Index: ep=1 → {db1:.4f}   ep=50 → {db50:.4f}"
      f"   Δ = {(db50-db1)/db1*100:+.1f}%")

# ─── figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)

data = [
    (axes[0], emb1,  labels1,  db1,  "(a) Instantaneous Freq. Space (Epoch 1)"),
    (axes[1], emb50, labels50, db50, "(b) Instantaneous Freq. Space (Epoch 50)"),
]

for ax, emb, labels, db, title in data:
    for cls_id in sorted(CLASS_NAMES):
        mask = labels == cls_id
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=CLASS_COLORS[cls_id],
                   marker=CLASS_MARKERS[cls_id],
                   label=CLASS_NAMES[cls_id],
                   s=12, alpha=0.70, linewidths=0.2, edgecolors="white")
    ax.set_title(f"{title}\nDavies-Bouldin Index = {db:.2f}", pad=4)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

# legend on right panel only
axes[1].legend(loc="lower right", markerscale=1.8, framealpha=0.85,
               handletextpad=0.4, borderpad=0.5)

# DB delta annotation on right panel
delta_pct = (db50 - db1) / db1 * 100
axes[1].text(0.03, 0.97, f"ΔDB = {delta_pct:+.1f}%  (↓ = more organized)",
             transform=axes[1].transAxes, fontsize=7.5,
             va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

fig.suptitle(
    "BearMamba-3 / CWRU SNR=0 dB / L_kin (λ=0.01, cover) / 5 seeds\n"
    "f_bar ∈ R²⁵⁶ = time-averaged instantaneous state frequencies",
    fontsize=8, color="#444444"
)

out_stem = "tsne_fbar_cwru_snr0"
fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=200)
print(f"Saved → {OUT_DIR / out_stem}.{{pdf,png}}")
plt.close(fig)
