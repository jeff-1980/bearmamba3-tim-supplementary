#!/usr/bin/env python3
"""
Step 5 / t-SNE: Pre-classifier feature space visualization.

Trains BearMamba-3 with L_kin (kin) and without (nokin) for 50 epochs on
CWRU SNR=0 dB (4-class, seed=2), then extracts pre-classifier features h∈R^64,
runs t-SNE, computes Davies-Bouldin Index, and plots a 2-panel comparison figure.

Pre-classifier feature: h = LayerNorm(hidden).mean(time)  [shape (B, d_model=64)]
captured via register_forward_hook on model.classifier.

Usage:
    source venv/bin/activate
    python experiments/step5_tsne.py [--snr 0] [--seed 2]
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bearmamba3.data_cwru import CWRUDataset
from bearmamba3.kinematic_loss import kinematic_loss, compute_fault_freqs
from bearmamba3.model import BearMamba3

# ─── config ───────────────────────────────────────────────────────────────────
DATA_DIR   = PROJECT_ROOT / "data" / "cwru_12k_de"
OUT_DIR    = PROJECT_ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES  = {0: "Normal", 1: "Inner", 2: "Ball", 3: "Outer"}
CLASS_COLORS = {0: "#999999", 1: "#2ca02c", 2: "#ff7f0e", 3: "#1f77b4"}

MODEL_KW = dict(d_model=64, d_state=128, n_layers=4, n_sensors=1,
                n_classes=4, conv_stride=2, is_mimo=False,
                mimo_rank=4, rope_fraction=0.5, dtype=torch.bfloat16)

TRAIN_KW = dict(lr=3e-4, weight_decay=1e-4, grad_clip=1.0, epochs=50,
                batch_size=64, val_ratio=0.2)

BEARING_KW = dict()   # CWRU — default params in compute_fault_freqs


# ─── helpers ──────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


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
    kw = dict(num_workers=4, pin_memory=True)
    tr_loader  = DataLoader(Subset(ds, tr_idx),
                            batch_size=TRAIN_KW["batch_size"], shuffle=True,  **kw)
    val_loader = DataLoader(Subset(ds, val_idx),
                            batch_size=256, shuffle=False, **kw)
    return tr_loader, val_loader


def build_model(device):
    kw = {k: v for k, v in MODEL_KW.items() if k != "mime_rank"}
    model = BearMamba3(**kw).to(device)
    return model


def train_one(model, tr_loader, val_loader, lambda_kin, device, fs_eff):
    opt = torch.optim.AdamW(model.parameters(),
                            lr=TRAIN_KW["lr"], weight_decay=TRAIN_KW["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=TRAIN_KW["epochs"])
    ce = nn.CrossEntropyLoss()

    for ep in range(1, TRAIN_KW["epochs"] + 1):
        model.train()
        for x, labels, rpm in tr_loader:
            x, labels, rpm = x.to(device), labels.to(device), rpm.to(device)
            if lambda_kin > 0:
                logits, kin = model(x.unsqueeze(1) if x.dim() == 2 else x,
                                    return_kin=True)
                loss = ce(logits, labels) + lambda_kin * kinematic_loss(
                    kin, rpm, fs_eff, variant="cover")
            else:
                logits = model(x.unsqueeze(1) if x.dim() == 2 else x)
                loss = ce(logits, labels)
            opt.zero_grad(); loss.backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_KW["grad_clip"])
            opt.step()
        scheduler.step()

        if ep % 10 == 0 or ep == TRAIN_KW["epochs"]:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, labels, _ in val_loader:
                    x, labels = x.to(device), labels.to(device)
                    logits = model(x.unsqueeze(1) if x.dim() == 2 else x)
                    correct += (logits.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            acc = correct / total * 100
            lbl = "kin" if lambda_kin > 0 else "nokin"
            print(f"  [{lbl}] ep={ep:3d}  val_acc={acc:.2f}%")


@torch.no_grad()
def extract_features(model, loader, device):
    """Return (features, labels) from val_loader using hook on model.classifier."""
    feats_list, labels_list = [], []
    handles = []

    def hook(module, inp, out):
        feats_list.append(inp[0].float().cpu().numpy())

    handles.append(model.classifier.register_forward_hook(hook))

    model.eval()
    for x, labels, _ in loader:
        x = x.to(device)
        _ = model(x.unsqueeze(1) if x.dim() == 2 else x)
        labels_list.append(labels.numpy())

    for h in handles:
        h.remove()

    return (np.concatenate(feats_list, axis=0),
            np.concatenate(labels_list, axis=0))


def run_tsne(feats: np.ndarray, seed: int = 42):
    scaler = StandardScaler()
    feats_s = scaler.fit_transform(feats)
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000,
                random_state=seed, init="pca", learning_rate="auto")
    return tsne.fit_transform(feats_s), feats_s   # (N,2), scaled feats for DB Index


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr",  type=float, default=0.0,  help="SNR dB (None=clean)")
    ap.add_argument("--seed", type=int,   default=2,    help="random seed")
    args = ap.parse_args()

    snr_db   = args.snr
    seed     = args.seed
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs_eff   = 12000 / 2   # CWRU 12kHz, conv_stride=2 → 6000 Hz

    set_seed(seed)
    tr_loader, val_loader = build_loaders(snr_db, seed)
    print(f"Dataset: {len(tr_loader.dataset)} train  {len(val_loader.dataset)} val")
    print(f"SNR={snr_db} dB  seed={seed}  device={device}")

    results = {}
    for label, lk in [("kin", 0.01), ("nokin", 0.0)]:
        print(f"\n── Training {label} (λ={lk}) ──")
        set_seed(seed)
        model = build_model(device)
        train_one(model, tr_loader, val_loader, lk, device, fs_eff)

        print(f"  Extracting features …")
        feats, labels = extract_features(model, val_loader, device)
        print(f"  feats shape: {feats.shape}  labels unique: {np.unique(labels)}")

        print(f"  Running t-SNE …")
        emb_2d, feats_scaled = run_tsne(feats, seed=42)
        db = davies_bouldin_score(feats_scaled, labels)
        print(f"  DB Index = {db:.4f}")

        results[label] = dict(emb=emb_2d, labels=labels, db=db)
        del model
        torch.cuda.empty_cache()

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)

    for ax, key, title in zip(axes,
                               ["nokin", "kin"],
                               ["(a) Without L_kin", "(b) With L_kin (λ=0.01, cover)"]):
        emb    = results[key]["emb"]
        labels = results[key]["labels"]
        db     = results[key]["db"]

        for cls_id, cls_name in CLASS_NAMES.items():
            mask = labels == cls_id
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=CLASS_COLORS[cls_id], label=cls_name,
                       s=8, alpha=0.65, linewidths=0)

        ax.set_title(f"{title}\nDavies-Bouldin Index = {db:.3f}", pad=4)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.tick_params(left=False, bottom=False,
                       labelleft=False, labelbottom=False)
        if key == "kin":
            ax.legend(loc="lower right", markerscale=2.5, framealpha=0.85)

    fig.suptitle(
        f"BearMamba-3 pre-classifier features / CWRU SNR={snr_db} dB / seed={seed}",
        fontsize=8, color="#444444"
    )

    out_stem = f"tsne_cwru_snr{int(snr_db):+d}_seed{seed}"
    fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=200)
    print(f"\nSaved → {OUT_DIR / out_stem}.{{pdf,png}}")

    print(f"\nDB Index summary:")
    print(f"  nokin: {results['nokin']['db']:.4f}")
    print(f"  kin:   {results['kin']['db']:.4f}")
    diff = results['nokin']['db'] - results['kin']['db']
    print(f"  Δ (nokin−kin): {diff:+.4f}  "
          f"({'kin better ✓' if diff > 0 else 'nokin better ⚠️'})")

    plt.close(fig)


if __name__ == "__main__":
    main()
