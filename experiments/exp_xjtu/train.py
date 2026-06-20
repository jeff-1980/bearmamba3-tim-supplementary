"""
experiments/exp_xjtu/train.py

XJTU-SY BearMamba-3 训练脚本 (D26)
支持两种模式:
  - lobo:  Cond3 LOBO（4折 × 5 seeds），指标 per-bearing recall → macro recall
  - cross: Cond2→Cond3 跨工况（5 seeds），OR:IR 类别权重，指标 macro-F1 + per-class recall

⚠️  config.yaml 中必须有 fs_eff: 12800（fs=25600 / conv_stride=2），脚本内含 assert。

用法:
  source venv/bin/activate
  python experiments/exp_xjtu/train.py --config config_lobo_nokin.yaml [--smoke]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bearmamba3.data_xjtu import (
    XJTUDataset, make_lobo_folds, make_cross_condition_split,
    valid_bearings, BEARING_FAILURE, CONDITION_RPM, BEARING_CONDITION,
)
from bearmamba3.kinematic_loss import kinematic_loss, instantaneous_freqs, compute_fault_freqs
from bearmamba3.model import BearMamba3

# Cond3 LOBO bearings (D26)
COND3_LOBO_BEARINGS = ['Bearing3_1', 'Bearing3_3', 'Bearing3_4', 'Bearing3_5']


# ── Utilities ────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("data_root", "results_dir"):
        if key in cfg:
            cfg[key] = str(Path(cfg[key]).expanduser())
    return cfg


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    backbone = cfg.get("backbone", "mamba3")
    if backbone == "cnn1d":
        from baselines.cnn1d import BearCNN1D
        return BearCNN1D(
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_sensors=cfg.get("n_sensors", 1),
            n_classes=cfg.get("n_classes", 2),
            conv_stride=cfg["conv_stride"],
        ).to(device)
    return BearMamba3(
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        n_layers=cfg["n_layers"],
        n_sensors=cfg.get("n_sensors", 1),
        n_classes=cfg.get("n_classes", 2),
        conv_stride=cfg["conv_stride"],
        is_mimo=False,
        use_batchnorm=cfg.get("use_batchnorm", False),
        dtype=torch.bfloat16,
    ).to(device)


def make_weighted_loader(dataset: XJTUDataset, batch_size: int,
                         num_workers: int, smoke: bool,
                         use_class_weights: bool = False):
    if use_class_weights:
        labels = np.array(dataset._labels)
        class_counts = np.maximum(
            np.bincount(labels, minlength=2).astype(np.float64), 1.0
        )
        sample_weights = (len(labels) / (2.0 * class_counts))[labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(dataset),
            replacement=True,
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            num_workers=0 if smoke else num_workers,
            pin_memory=True, drop_last=True,
        )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0 if smoke else num_workers,
        pin_memory=True, drop_last=True,
    )


def eval_recall_f1(model, loader, device, n_classes: int = 2):
    """Return per-class recall, macro recall, macro F1."""
    model.eval()
    tp = np.zeros(n_classes, dtype=int)
    fn = np.zeros(n_classes, dtype=int)
    fp = np.zeros(n_classes, dtype=int)
    with torch.no_grad():
        for x, labels, _ in loader:
            x, labels = x.to(device), labels.to(device)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            preds = logits.argmax(1).cpu().numpy()
            labs  = labels.cpu().numpy()
            for c in range(n_classes):
                tp[c] += ((preds == c) & (labs == c)).sum()
                fn[c] += ((preds != c) & (labs == c)).sum()
                fp[c] += ((preds == c) & (labs != c)).sum()
    recall    = tp / np.maximum(tp + fn, 1).astype(float)
    precision = tp / np.maximum(tp + fp, 1).astype(float)
    f1_per    = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
    macro_recall = recall.mean()
    macro_f1     = f1_per.mean()
    return recall, macro_recall, macro_f1


def save_kin_snapshot(model, snap_batch, fs_eff: float, epoch: int,
                      tag: str, results_dir: str, device,
                      bearing_kwargs: dict | None = None):
    x, labels, rpm = [t.to(device) for t in snap_batch]
    model.eval()
    with torch.no_grad():
        _, kin = model(x, return_kin=True)
    f_bar   = instantaneous_freqs(kin, fs_eff).mean(dim=-1).cpu().numpy()
    f_fault = compute_fault_freqs(rpm, device=device, **(bearing_kwargs or {})).cpu().numpy()
    path = Path(results_dir) / f"{tag}_kin_ep{epoch:03d}.npz"
    np.savez_compressed(
        path,
        f_bar=f_bar.astype(np.float32),
        f_fault=f_fault.astype(np.float32),
        labels=labels.cpu().numpy().astype(np.int32),
        epoch=np.int32(epoch),
        fs_eff=np.float32(fs_eff),
    )


# ── Per-fold / per-seed training ─────────────────────────────────────────────

def train_one_run(
    cfg: dict,
    seed: int,
    train_ds: XJTUDataset,
    test_ds: XJTUDataset,
    device: torch.device,
    smoke: bool,
    fold_tag: str,
    use_class_weights: bool = False,
) -> dict:
    set_seed(seed)

    bs         = cfg["batch_size"]
    nw         = cfg.get("num_workers", 4)
    lambda_kin = cfg.get("lambda_kin", 0.0)
    kin_variant= cfg.get("kin_variant", "cover")
    bkw        = cfg.get("bearing_kwargs", {})
    fs_eff     = float(cfg["fs_eff"])
    n_epochs   = 2 if smoke else cfg["epochs"]
    results_dir= cfg["results_dir"]

    train_loader = make_weighted_loader(train_ds, bs, nw, smoke, use_class_weights)
    test_loader  = torch.utils.data.DataLoader(
        test_ds, batch_size=bs * 2, shuffle=False,
        num_workers=0 if smoke else nw, pin_memory=True,
    )

    model     = build_model(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    backbone = cfg.get("backbone", "mamba3")
    do_kin = (lambda_kin > 0) and (backbone not in ("cnn1d", "transformer1d"))

    # I1 snapshot setup (for L_kin runs)
    snap_batch  = None
    snap_tag    = None
    snap_epochs = {1} | {e for e in range(10, n_epochs + 1, 10)}
    if do_kin and not smoke:
        snap_batch = next(iter(test_loader))
        lk_str = f"lk{lambda_kin:.0e}".replace("e-0","e-").replace("e+0","e")
        snap_tag = f"{cfg.get('name','exp')}_{fold_tag}_seed{seed}_{lk_str}"

    best_macro_recall = 0.0
    best_macro_f1     = 0.0
    best_per_class_recall = np.zeros(2)
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        running_ce = running_kin = n_steps = 0

        for step, (x, labels, rpm) in enumerate(train_loader):
            if smoke and step >= 2:
                break
            x, labels, rpm = x.to(device), labels.to(device), rpm.to(device)

            if do_kin:
                out, kin = model(x, return_kin=True)
                l_ce  = nn.functional.cross_entropy(out, labels)
                l_kin = kinematic_loss(kin, rpm, fs_eff,
                                       variant=kin_variant, bearing_kwargs=bkw)
                loss  = l_ce + lambda_kin * l_kin
            else:
                out  = model(x)
                l_ce = nn.functional.cross_entropy(out, labels)
                l_kin = torch.tensor(0.0)
                loss  = l_ce

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
            optimizer.step()

            running_ce  += l_ce.item()
            running_kin += l_kin.item()
            n_steps += 1

        scheduler.step()

        recall, macro_recall, macro_f1 = eval_recall_f1(model, test_loader, device)
        best_macro_recall = max(best_macro_recall, macro_recall)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_per_class_recall = recall.copy()

        if snap_batch is not None and epoch in snap_epochs:
            save_kin_snapshot(model, snap_batch, fs_eff, epoch,
                              snap_tag, results_dir, device, bearing_kwargs=bkw)

        if epoch % 10 == 0 or epoch <= 3 or smoke:
            elapsed = time.time() - t0
            print(
                f"  {fold_tag} seed={seed} ep={epoch:3d}/{n_epochs}"
                f"  l_ce={running_ce/max(n_steps,1):.4f}"
                f"  l_kin={running_kin/max(n_steps,1):.4f}"
                f"  recall=[{recall[0]:.3f},{recall[1]:.3f}]"
                f"  macro_recall={macro_recall:.4f}  [{time.time()-t0:.0f}s]"
            )

    return {
        "fold_tag": fold_tag,
        "seed": seed,
        "best_macro_recall": float(best_macro_recall),
        "best_macro_f1": float(best_macro_f1),
        "best_per_class_recall": best_per_class_recall.tolist(),
        "elapsed_s": time.time() - t0,
    }


# ── LOBO mode ─────────────────────────────────────────────────────────────────

def run_lobo(cfg: dict, device: torch.device, smoke: bool):
    data_root   = cfg["data_root"]
    seeds       = cfg["seeds"]
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    folds = make_lobo_folds(COND3_LOBO_BEARINGS)  # 4 folds
    all_seed_macro = []  # per-seed macro recalls

    for seed in seeds:
        seed_fold_results = []
        or_recalls, ir_recalls = [], []

        for fold_idx, (train_bearings, test_bearings) in enumerate(folds):
            test_bearing = test_bearings[0]
            test_label   = BEARING_FAILURE[test_bearing]  # 'OR' or 'IR'
            fold_tag     = f"fold{fold_idx}_{test_bearing}"

            out_path = results_dir / f"fold{fold_idx}_seed{seed}.json"
            if out_path.exists():
                print(f"  [SKIP] {out_path.name}")
                with open(out_path) as f:
                    res = json.load(f)
                seed_fold_results.append(res)
                recall_val = res["best_per_class_recall"][
                    0 if test_label == 'OR' else 1
                ]
                if test_label == 'OR':
                    or_recalls.append(recall_val)
                else:
                    ir_recalls.append(recall_val)
                continue

            print(f"\n{'='*60}")
            print(f"Fold {fold_idx}: train={train_bearings} / test={test_bearing}({test_label})")

            n_sensors = cfg.get("n_sensors", 1)
            train_ds = XJTUDataset(data_root, train_bearings, n_sensors=n_sensors)
            test_ds  = XJTUDataset(data_root, test_bearings,  n_sensors=n_sensors)
            if smoke:
                # Tiny subset for smoke test
                train_ds._windows = train_ds._windows[:128]
                train_ds._labels  = train_ds._labels[:128]
                train_ds._rpms    = train_ds._rpms[:128]
                test_ds._windows  = test_ds._windows[:64]
                test_ds._labels   = test_ds._labels[:64]
                test_ds._rpms     = test_ds._rpms[:64]

            res = train_one_run(cfg, seed, train_ds, test_ds, device, smoke, fold_tag)
            seed_fold_results.append(res)

            with open(out_path, "w") as f:
                json.dump(res, f, indent=2)
            print(f"  → saved {out_path}")

            # The test bearing is single-class: recall is the relevant class recall
            # recall[0]=OR recall, recall[1]=IR recall
            recall_val = res["best_per_class_recall"][0 if test_label == 'OR' else 1]
            if test_label == 'OR':
                or_recalls.append(recall_val)
            else:
                ir_recalls.append(recall_val)

        # Aggregate this seed's macro recall
        if or_recalls and ir_recalls:
            macro = (np.mean(or_recalls) + np.mean(ir_recalls)) / 2
        elif or_recalls:
            macro = np.mean(or_recalls)
        else:
            macro = np.mean(ir_recalls)

        all_seed_macro.append(float(macro))
        print(f"\nSeed {seed}: OR_recall={or_recalls}  IR_recall={ir_recalls}  macro={macro:.4f}")

    # Save summary
    mean_macro = float(np.mean(all_seed_macro))
    std_macro  = float(np.std(all_seed_macro, ddof=1)) if len(all_seed_macro) > 1 else 0.0
    summary = {
        "mode": "lobo",
        "config": cfg.get("name"),
        "lambda_kin": cfg.get("lambda_kin", 0.0),
        "lobo_bearings": COND3_LOBO_BEARINGS,
        "seeds": seeds,
        "per_seed_macro_recall": all_seed_macro,
        "mean_macro_recall": mean_macro,
        "std_macro_recall": std_macro,
    }
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f"LOBO RESULTS: macro_recall = {mean_macro*100:.2f}±{std_macro*100:.2f}%")
    print(f"Per-seed: {[f'{v*100:.2f}' for v in all_seed_macro]}")
    print(f"Summary saved to {summary_path}")


# ── Cross-condition mode ──────────────────────────────────────────────────────

def run_cross(cfg: dict, device: torch.device, smoke: bool):
    data_root   = cfg["data_root"]
    seeds       = cfg["seeds"]
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    train_cond = cfg.get("train_condition", "37.5Hz11kN")
    test_cond  = cfg.get("test_condition",  "40Hz10kN")
    train_bearings, test_bearings = make_cross_condition_split(train_cond, test_cond)

    print(f"Cross-condition: {train_cond} → {test_cond}")
    print(f"Train: {train_bearings}")
    print(f"Test:  {test_bearings}")

    # Class distribution in training set
    n_sensors = cfg.get("n_sensors", 1)
    train_ds_full = XJTUDataset(data_root, train_bearings, n_sensors=n_sensors)
    labels_arr = np.array(train_ds_full._labels)
    counts = np.bincount(labels_arr, minlength=2)
    print(f"Train class counts: OR={counts[0]}, IR={counts[1]}, ratio={counts[0]/max(counts[1],1):.1f}:1")

    test_ds = XJTUDataset(data_root, test_bearings, n_sensors=n_sensors)
    if smoke:
        train_ds_full._windows = train_ds_full._windows[:128]
        train_ds_full._labels  = train_ds_full._labels[:128]
        train_ds_full._rpms    = train_ds_full._rpms[:128]
        test_ds._windows = test_ds._windows[:64]
        test_ds._labels  = test_ds._labels[:64]
        test_ds._rpms    = test_ds._rpms[:64]

    all_results = []
    for seed in seeds:
        out_path = results_dir / f"seed_{seed}.json"
        if out_path.exists():
            print(f"  [SKIP] {out_path.name}")
            with open(out_path) as f:
                all_results.append(json.load(f))
            continue

        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        res = train_one_run(cfg, seed, train_ds_full, test_ds,
                            device, smoke, f"cross_seed{seed}",
                            use_class_weights=True)
        res["train_class_counts"] = counts.tolist()
        all_results.append(res)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2)
        print(f"  → saved {out_path}")

    # Summary
    macro_f1s     = [r["best_macro_f1"] for r in all_results]
    macro_recalls = [r["best_macro_recall"] for r in all_results]
    mean_f1  = float(np.mean(macro_f1s))
    std_f1   = float(np.std(macro_f1s, ddof=1)) if len(macro_f1s) > 1 else 0.0
    mean_rec = float(np.mean(macro_recalls))
    std_rec  = float(np.std(macro_recalls, ddof=1)) if len(macro_recalls) > 1 else 0.0
    summary = {
        "mode": "cross",
        "config": cfg.get("name"),
        "lambda_kin": cfg.get("lambda_kin", 0.0),
        "train_condition": train_cond,
        "test_condition":  test_cond,
        "seeds": seeds,
        "macro_f1s": macro_f1s,
        "macro_recalls": macro_recalls,
        "mean_macro_f1": mean_f1,
        "std_macro_f1":  std_f1,
        "mean_macro_recall": mean_rec,
        "std_macro_recall":  std_rec,
    }
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*60}")
    print(f"CROSS-CONDITION RESULTS: macro_F1 = {mean_f1*100:.2f}±{std_f1*100:.2f}%")
    print(f"                         macro_recall = {mean_rec*100:.2f}±{std_rec*100:.2f}%")
    print(f"Per-seed F1:  {[f'{v*100:.2f}' for v in macro_f1s]}")
    print(f"Summary saved to {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_lobo_nokin.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cwd_rel = Path.cwd() / args.config
        cfg_path = cwd_rel if cwd_rel.exists() else Path(__file__).parent / args.config
    cfg = load_config(cfg_path)
    if args.seeds:
        cfg["seeds"] = args.seeds

    # ⚠️ fs_eff sanity check (D26 mandatory)
    assert "fs_eff" in cfg, "config.yaml must contain fs_eff (expected 12800 for XJTU)"
    assert abs(cfg["fs_eff"] - 12800.0) < 1, \
        f"fs_eff={cfg['fs_eff']} — expected 12800 (fs=25600 / conv_stride=2)"

    mode = cfg.get("mode", "lobo")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  mode={mode}  backbone={cfg.get('backbone','mamba3')}"
          f"  lambda_kin={cfg.get('lambda_kin',0)}  fs_eff={cfg['fs_eff']}"
          f"  seeds={cfg['seeds']}  epochs={2 if args.smoke else cfg['epochs']}")

    if mode == "lobo":
        run_lobo(cfg, device, args.smoke)
    elif mode == "cross":
        run_cross(cfg, device, args.smoke)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'lobo' or 'cross'.")


if __name__ == "__main__":
    main()
