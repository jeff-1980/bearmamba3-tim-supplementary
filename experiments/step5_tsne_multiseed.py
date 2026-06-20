#!/usr/bin/env python3
"""
step5_tsne_multiseed.py — Multi-seed t-SNE composite for BearMamba-3 C2/C3 robustness.

Layout: 2 rows × 5 cols
  Row 0: SNR=0 dB,   seeds {0,1,2,3,4}  (+L_kin, λ=0.01)
  Row 1: SNR=−6 dB,  seeds {0,1,2,3,4}  (+L_kin, λ=0.01)

Each subplot title: "seed N  DB={value:.3f}"
Row-right annotation: "mean DB ± std"

Features: pre-classifier h ∈ R^64 (same as step5_tsne.py, via register_forward_hook).
Cache:    results/tsne_feats/  — saved as .npz after each seed/SNR so runs resume safely.

Usage:
    source venv/bin/activate
    python experiments/step5_tsne_multiseed.py [--force-retrain]
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import StandardScaler

# ── matplotlib style (DejaVu Serif, pdf.fonttype=42, matching visualize_i1.py) ──
plt.rcParams.update({
    "font.family":      "DejaVu Serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bearmamba3.data_cwru import CWRUDataset
from bearmamba3.kinematic_loss import kinematic_loss, compute_fault_freqs
from bearmamba3.model import BearMamba3

# ── config ────────────────────────────────────────────────────────────────────
DATA_DIR  = PROJECT_ROOT / "data" / "cwru_12k_de"
OUT_DIR   = PROJECT_ROOT / "results" / "figures"
CACHE_DIR = PROJECT_ROOT / "results" / "tsne_feats"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEEDS    = [0, 1, 2, 3, 4]
SNR_LIST = [0.0, -6.0]       # row 0 = SNR=0, row 1 = SNR=-6
LAMBDA_KIN = 0.01

CLASS_NAMES  = {0: "Normal", 1: "Inner", 2: "Ball", 3: "Outer"}
CLASS_COLORS = {0: "#999999", 1: "#2ca02c", 2: "#ff7f0e", 3: "#1f77b4"}

MODEL_KW = dict(d_model=64, d_state=128, n_layers=4, n_sensors=1,
                n_classes=4, conv_stride=2, is_mimo=False,
                mimo_rank=4, rope_fraction=0.5, dtype=torch.bfloat16)

TRAIN_KW = dict(lr=3e-4, weight_decay=1e-4, grad_clip=1.0, epochs=50,
                batch_size=64, val_ratio=0.2, num_workers=4)

FS_EFF = 12000 / 2   # CWRU 12kHz, conv_stride=2 → 6000 Hz


# ── helpers ───────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(snr_db, seed):
    ds = CWRUDataset(DATA_DIR, win_len=2048, stride=1024,
                     channels=["DE"], normalize=True,
                     seed=seed, noise_snr_db=snr_db,
                     label_mode="4class")
    n_val   = int(len(ds) * TRAIN_KW["val_ratio"])
    n_train = len(ds) - n_val
    idx = list(range(len(ds)))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    tr_idx, val_idx = idx[:n_train], idx[n_train:]
    kw = dict(num_workers=TRAIN_KW["num_workers"], pin_memory=True)
    tr_loader  = DataLoader(Subset(ds, tr_idx),
                            batch_size=TRAIN_KW["batch_size"],
                            shuffle=True, **kw)
    val_loader = DataLoader(Subset(ds, val_idx),
                            batch_size=256, shuffle=False, **kw)
    return tr_loader, val_loader


def build_model(device):
    return BearMamba3(**MODEL_KW).to(device)


def train_kin(model, tr_loader, val_loader, device, seed):
    opt = torch.optim.AdamW(model.parameters(),
                            lr=TRAIN_KW["lr"],
                            weight_decay=TRAIN_KW["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=TRAIN_KW["epochs"])
    ce = nn.CrossEntropyLoss()

    for ep in range(1, TRAIN_KW["epochs"] + 1):
        model.train()
        for x, labels, rpm in tr_loader:
            x, labels, rpm = x.to(device), labels.to(device), rpm.to(device)
            logits, kin = model(x.unsqueeze(1) if x.dim() == 2 else x,
                                return_kin=True)
            loss = ce(logits, labels) + LAMBDA_KIN * kinematic_loss(
                kin, rpm, FS_EFF, variant="cover")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_KW["grad_clip"])
            opt.step()
        scheduler.step()

    # final val acc
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, lbl, _ in val_loader:
            x, lbl = x.to(device), lbl.to(device)
            logits = model(x.unsqueeze(1) if x.dim() == 2 else x)
            correct += (logits.argmax(1) == lbl).sum().item()
            total   += lbl.size(0)
    return correct / total * 100


@torch.no_grad()
def extract_features(model, loader, device):
    """Pre-classifier h ∈ R^64 via forward hook on model.classifier."""
    feats_list, labels_list = [], []

    def hook(module, inp, out):
        feats_list.append(inp[0].float().cpu().numpy())

    h = model.classifier.register_forward_hook(hook)
    model.eval()
    for x, lbl, _ in loader:
        x = x.to(device)
        _ = model(x.unsqueeze(1) if x.dim() == 2 else x)
        labels_list.append(lbl.numpy())
    h.remove()

    return (np.concatenate(feats_list, axis=0),
            np.concatenate(labels_list, axis=0))


def run_tsne(feats: np.ndarray):
    scaler = StandardScaler()
    feats_s = scaler.fit_transform(feats)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                random_state=42, init="pca", learning_rate="auto")
    emb_2d = tsne.fit_transform(feats_s)
    return emb_2d, feats_s


def cache_path(snr_db, seed):
    snr_tag = f"snr{int(snr_db):+d}"
    return CACHE_DIR / f"feats_{snr_tag}_seed{seed}.npz"


def get_or_compute(snr_db, seed, device, force_retrain):
    cpath = cache_path(snr_db, seed)
    if cpath.exists() and not force_retrain:
        print(f"  [cache hit] {cpath.name}")
        d = np.load(cpath)
        return d["feats"], d["labels"]

    print(f"  [train]  SNR={snr_db:+.0f}dB  seed={seed}  λ={LAMBDA_KIN}")
    t0 = time.time()
    set_seed(seed)
    tr_loader, val_loader = build_loaders(snr_db, seed)
    model = build_model(device)
    val_acc = train_kin(model, tr_loader, val_loader, device, seed)
    feats, labels = extract_features(model, val_loader, device)
    elapsed = time.time() - t0
    print(f"    val_acc={val_acc:.2f}%  elapsed={elapsed:.0f}s  "
          f"feats={feats.shape}")
    del model
    torch.cuda.empty_cache()

    np.savez_compressed(cpath, feats=feats, labels=labels)
    print(f"    → cached {cpath}")
    return feats, labels


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-retrain", action="store_true",
                    help="Ignore cached features and retrain all seeds")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── collect all data ──────────────────────────────────────────────────────
    # results[snr_idx][seed_idx] = {"emb": ..., "labels": ..., "db": ...}
    results = {}
    for ri, snr_db in enumerate(SNR_LIST):
        results[ri] = {}
        snr_str = f"SNR={snr_db:+.0f} dB"
        print(f"\n{'='*50}")
        print(f"  Row {ri}: {snr_str}")
        print(f"{'='*50}")
        for ci, seed in enumerate(SEEDS):
            print(f"\n  seed={seed}")
            feats, labels = get_or_compute(snr_db, seed, device,
                                           args.force_retrain)
            print(f"  Running t-SNE …")
            emb_2d, feats_s = run_tsne(feats)
            db = davies_bouldin_score(feats_s, labels)
            print(f"  DB-Index = {db:.4f}")
            results[ri][ci] = dict(emb=emb_2d, labels=labels, db=db,
                                   seed=seed)

    # ── compute row stats ─────────────────────────────────────────────────────
    row_stats = {}
    for ri in range(len(SNR_LIST)):
        dbs = [results[ri][ci]["db"] for ci in range(len(SEEDS))]
        row_stats[ri] = (np.mean(dbs), np.std(dbs, ddof=1))

    # ── plot 2×5 grid ─────────────────────────────────────────────────────────
    ncols = len(SEEDS)
    nrows = len(SNR_LIST)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(13.0, 5.4),
                             constrained_layout=True)

    for ri, snr_db in enumerate(SNR_LIST):
        for ci, seed in enumerate(SEEDS):
            ax  = axes[ri, ci]
            res = results[ri][ci]
            emb    = res["emb"]
            labels = res["labels"]
            db     = res["db"]

            for cls_id, cls_name in CLASS_NAMES.items():
                mask = labels == cls_id
                if mask.sum() == 0:
                    continue
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           c=CLASS_COLORS[cls_id],
                           label=cls_name,
                           s=5, alpha=0.60, linewidths=0)

            ax.set_title(f"seed {seed}\nDB={db:.3f}", pad=3)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

            # legend only on top-right subplot
            if ri == 0 and ci == ncols - 1:
                ax.legend(loc="lower right", markerscale=2.0,
                          framealpha=0.85, fontsize=7,
                          handletextpad=0.3, borderpad=0.4)

        # row label on rightmost subplot
        mu, sd = row_stats[ri]
        axes[ri, -1].annotate(
            f"SNR={snr_db:+.0f} dB\nmean DB={mu:.3f}±{sd:.3f}",
            xy=(1.03, 0.5), xycoords="axes fraction",
            va="center", ha="left", fontsize=7.5,
            color="#333333",
        )

    fig.suptitle(
        "BearMamba-3 (+L_kin) pre-classifier features — CWRU 4-class, all 5 seeds",
        fontsize=8, color="#444444", y=1.01,
    )

    out_stem = OUT_DIR / "tsne_multiseed_composite"
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{out_stem}.png", bbox_inches="tight", dpi=300)
    print(f"\nSaved → {out_stem}.{{pdf,png}}")
    plt.close(fig)

    # ── report ────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("DB-Index Report")
    print("="*60)
    for ri, snr_db in enumerate(SNR_LIST):
        dbs = [results[ri][ci]["db"] for ci in range(len(SEEDS))]
        mu, sd = row_stats[ri]
        print(f"\nSNR={snr_db:+.0f} dB:")
        for ci, (seed, db) in enumerate(zip(SEEDS, dbs)):
            print(f"  seed {seed}: DB={db:.4f}")
        print(f"  → mean={mu:.4f}  std={sd:.4f}  "
              f"{'stable ✓' if sd < 0.1 else 'high variance ⚠'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
