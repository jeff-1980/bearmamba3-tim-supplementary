"""
experiments/exp06_pu/train.py

PU (Paderborn) 3类消融 — BearMamba3/BearMamba2 SISO, CE-only 或 CE+L_kin
跨工况: 训练 N09_M07_F10 + N15_M01_F10, 测试 N15_M07_F10
M4 矩阵: BM3+nokin / BM3+kin / BM2+nokin（BM2 不支持 L_kin）

用法:
  source venv/bin/activate
  python experiments/exp06_pu/train.py [--config config.yaml] [--smoke]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bearmamba3.data_pu import (
    PUDataset, BEARING_KWARGS, COND_TRAIN, COND_TEST, FS
)
from bearmamba3.kinematic_loss import kinematic_loss, instantaneous_freqs, compute_fault_freqs
from bearmamba3.model import BearMamba3


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("data_dir", "results_dir"):
        if key in cfg:
            cfg[key] = str(Path(cfg[key]).expanduser())
    return cfg


def build_loaders(cfg: dict, seed: int, smoke: bool):
    noise_snr = cfg.get("noise_snr_db", None)
    win_len   = cfg["win_len"]
    stride    = cfg.get("stride", win_len)
    cross     = cfg.get("cross_condition", True)

    if cross:
        train_ds = PUDataset(cfg["data_dir"], conditions=COND_TRAIN,
                             win_len=win_len, stride=stride,
                             noise_snr_db=noise_snr)
        val_ds   = PUDataset(cfg["data_dir"], conditions=COND_TEST,
                             win_len=win_len, stride=stride,
                             noise_snr_db=noise_snr)
    else:
        full = PUDataset(cfg["data_dir"], conditions=None,
                         win_len=win_len, stride=stride,
                         noise_snr_db=noise_snr)
        n_val   = int(len(full) * cfg.get("val_ratio", 0.2))
        n_train = len(full) - n_val
        gen = torch.Generator().manual_seed(seed)
        train_ds, val_ds = torch.utils.data.random_split(
            full, [n_train, n_val], generator=gen
        )

    bs = cfg["batch_size"]
    nw = 0 if smoke else cfg.get("num_workers", 4)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs * 2, shuffle=False,
        num_workers=nw, pin_memory=True
    )
    return train_loader, val_loader


def build_model(cfg: dict, device: torch.device) -> nn.Module:
    backbone = cfg.get("backbone", "mamba3")
    if backbone == "mamba2":
        from baselines.mamba2 import BearMamba2
        return BearMamba2(
            d_model=cfg["d_model"],
            d_state=cfg["d_state"],
            n_layers=cfg["n_layers"],
            n_sensors=1,
            n_classes=3,
            conv_stride=cfg["conv_stride"],
            dtype=torch.bfloat16,
        ).to(device)
    if backbone == "cnn1d":
        from baselines.cnn1d import BearCNN1D
        return BearCNN1D(
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_sensors=1,
            n_classes=3,
            conv_stride=cfg["conv_stride"],
        ).to(device)
    return BearMamba3(
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        n_layers=cfg["n_layers"],
        n_sensors=1,
        n_classes=3,
        conv_stride=cfg["conv_stride"],
        is_mimo=False,
        dtype=torch.bfloat16,
    ).to(device)


def val_epoch(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, labels, _ in loader:
            x, labels = x.to(device), labels.to(device)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def save_kin_snapshot(model, fixed_batch, fs_eff, epoch, tag, results_dir, device):
    x, labels, rpm = [t.to(device) for t in fixed_batch]
    model.eval()
    with torch.no_grad():
        _, kin = model(x, return_kin=True)
    f_bar   = instantaneous_freqs(kin, fs_eff).mean(dim=-1).cpu().numpy()   # (B, S)
    f_fault = compute_fault_freqs(rpm, device=device, **BEARING_KWARGS).cpu().numpy()
    path = Path(results_dir) / f"{tag}_kin_ep{epoch:03d}.npz"
    np.savez_compressed(
        path,
        f_bar=f_bar.astype(np.float32),
        f_fault=f_fault.astype(np.float32),
        labels=labels.cpu().numpy().astype(np.int32),
        epoch=np.int32(epoch),
        fs_eff=np.float32(fs_eff),
    )


def train_one_seed(cfg: dict, seed: int, device: torch.device,
                   smoke: bool = False) -> dict:
    set_seed(seed)
    train_loader, val_loader = build_loaders(cfg, seed, smoke)
    model = build_model(cfg, device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    n_epochs  = 2 if smoke else cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    backbone   = cfg.get("backbone", "mamba3")
    lambda_kin = cfg.get("lambda_kin", 0.0)
    kin_variant = cfg.get("kin_variant", "cover")
    fs_hz      = cfg.get("fs_hz", float(FS))
    fs_eff     = fs_hz / cfg["conv_stride"]
    do_kin     = (lambda_kin > 0) and (backbone not in ("mamba2", "cnn1d", "transformer1d"))

    snap_batch = None
    snap_tag   = None
    if do_kin and not smoke:
        snap_batch = next(iter(val_loader))
        exp_name   = cfg.get("name", "exp")
        lk_str     = f"lk{lambda_kin:.0e}".replace("e-0", "e-").replace("e+0", "e")
        snap_tag   = f"{exp_name}_seed{seed}_{lk_str}"
    snap_epochs = {1} | {e for e in range(10, n_epochs + 1, 10)}

    results_dir = cfg["results_dir"]
    history     = []
    best_val    = 0.0
    t0          = time.time()

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
                l_kin = kinematic_loss(kin, rpm, fs_eff, variant=kin_variant,
                                       bearing_kwargs=BEARING_KWARGS)
                loss  = l_ce + lambda_kin * l_kin
            else:
                out   = model(x)
                l_ce  = nn.functional.cross_entropy(out, labels)
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
        val_acc   = val_epoch(model, val_loader, device)
        best_val  = max(best_val, val_acc)

        log = {
            "epoch":   epoch,
            "l_ce":    running_ce  / max(n_steps, 1),
            "l_kin":   running_kin / max(n_steps, 1),
            "val_acc": val_acc,
        }
        history.append(log)

        if snap_batch is not None and epoch in snap_epochs:
            save_kin_snapshot(model, snap_batch, fs_eff, epoch, snap_tag,
                              results_dir, device)

        if epoch % 10 == 0 or epoch <= 3 or smoke:
            elapsed = time.time() - t0
            print(
                f"  seed={seed} epoch={epoch:3d}/{n_epochs}"
                f"  l_ce={log['l_ce']:.4f}  l_kin={log['l_kin']:.4f}"
                f"  val_acc={val_acc:.4f}  [{elapsed:.0f}s]"
            )

    elapsed = time.time() - t0
    return {
        "seed":          seed,
        "best_val_acc":  best_val,
        "final_val_acc": history[-1]["val_acc"],
        "elapsed_s":     elapsed,
        "history":       history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke",  action="store_true")
    ap.add_argument("--seeds",  nargs="+", type=int)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cwd_rel  = Path.cwd() / args.config
        cfg_path = cwd_rel if cwd_rel.exists() else Path(__file__).parent / args.config
    cfg   = load_config(cfg_path)
    seeds = args.seeds or cfg["seeds"]

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = cfg.get("backbone", "mamba3")
    snr      = cfg.get("noise_snr_db", None)
    print(f"Device: {device}  backbone={backbone}  lambda_kin={cfg.get('lambda_kin',0)}"
          f"  snr={snr}dB  seeds={seeds}  epochs={2 if args.smoke else cfg['epochs']}")

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        result = train_one_seed(cfg, seed, device, smoke=args.smoke)
        all_results.append(result)

        seed_path = results_dir / f"seed_{seed}.json"
        with open(seed_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  → saved {seed_path}")

    if not args.smoke:
        accs     = [r["best_val_acc"] for r in all_results]
        mean_acc = float(np.mean(accs))
        std_acc  = float(np.std(accs, ddof=1))
        summary  = {
            "config":           cfg_path.name,
            "lambda_kin":       cfg.get("lambda_kin", 0.0),
            "noise_snr_db":     snr,
            "seeds":            seeds,
            "best_val_accs":    accs,
            "mean_best_val_acc":mean_acc,
            "std_best_val_acc": std_acc,
        }
        summary_path = results_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n{'='*60}")
        print(f"RESULTS: mean±std = {mean_acc*100:.2f}±{std_acc*100:.2f}%")
        print(f"Summary saved to {summary_path}")
    else:
        print("\n✅ Smoke test passed")


if __name__ == "__main__":
    main()
