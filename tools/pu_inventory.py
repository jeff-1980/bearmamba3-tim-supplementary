"""
tools/pu_inventory.py — Paderborn University KAt 数据集清单

打印: 条件 × 轴承组合, 窗口数统计, 故障频率验证
用法: python tools/pu_inventory.py
"""
import glob
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from bearmamba3.data_pu import (
    BEARING_KWARGS, COND_RPM, LABEL_MAP, LABEL_NAMES, FS
)
from bearmamba3.kinematic_loss import compute_fault_freqs

DATA_DIR = os.path.expanduser("~/data_pu")

# ─── bearing taxonomy ────────────────────────────────────────────────────────
DAMAGE_CATEGORY = {
    "K001": "Normal (healthy)",
    "K002": "Normal (healthy)",
    "KA04": "Outer race — artificial damage",
    "KA15": "Outer race — artificial damage",
    "KI01": "Inner race — artificial damage",
    "KI03": "Inner race — artificial damage",
    "KI05": "Inner race — artificial damage",
    "KB23": "Real accelerated damage (excluded from 3-class)",
    "KB24": "Real accelerated damage (excluded from 3-class)",
    "KB27": "Real accelerated damage (excluded from 3-class)",
}


def parse_fname(path):
    base = os.path.basename(path).replace(".mat", "")
    parts = base.split("_")
    if len(parts) < 5:
        return None, None, None
    return "_".join(parts[:3]), parts[3], int(parts[4])


def main():
    print("=" * 70)
    print("Paderborn University KAt Bearing Dataset — Inventory")
    print("=" * 70)

    print(f"\nData directory : {DATA_DIR}")
    all_mat = sorted(glob.glob(os.path.join(DATA_DIR, "*.mat")))
    print(f"Total .mat files: {len(all_mat)}")

    print(f"\n{'Bearing geometry (SKF 6203, official KI01.pdf)':}")
    print(f"  n_balls={BEARING_KWARGS['n_balls']}  "
          f"d={BEARING_KWARGS['d']} mm  "
          f"D={BEARING_KWARGS['D']} mm (pitch circle)  "
          f"α={BEARING_KWARGS['contact_angle_deg']}°")
    print(f"  Sampling rate: {FS} Hz  (official: 64,498 Hz)")

    # ── working conditions ────────────────────────────────────────────────────
    print(f"\n{'Condition':<20} {'RPM':>6}  {'Files':>5}  {'Bearings'}")
    print("-" * 60)
    cond_bear = defaultdict(set)
    cond_count = defaultdict(int)
    for path in all_mat:
        cond, bear, _ = parse_fname(path)
        if cond:
            cond_bear[cond].add(bear)
            cond_count[cond] += 1
    for cond in sorted(COND_RPM):
        rpm   = COND_RPM[cond]
        bears = sorted(cond_bear.get(cond, []))
        print(f"  {cond:<18} {rpm:>6}  {cond_count[cond]:>5}  {', '.join(bears)}")

    # ── bearing taxonomy ─────────────────────────────────────────────────────
    print(f"\n{'Bearing ID':<8}  {'3-class label':<10}  {'Category'}")
    print("-" * 65)
    for bid, cat in DAMAGE_CATEGORY.items():
        lbl = LABEL_MAP.get(bid)
        lname = f"{lbl}:{LABEL_NAMES[lbl]}" if lbl is not None else "excluded"
        print(f"  {bid:<6}  {lname:<12}  {cat}")

    # ── fault frequencies at each working condition ───────────────────────────
    import torch
    print("\nFault frequencies (Hz) per working condition:")
    print(f"  {'Condition':<20} {'fr':>7} {'FTF':>7} {'BPFO':>8} {'2xBSF':>8} {'BPFI':>8}")
    print("  " + "-" * 55)
    for cond, rpm in sorted(COND_RPM.items()):
        freqs = compute_fault_freqs(
            torch.tensor(float(rpm)), **BEARING_KWARGS, device="cpu"
        ).numpy()
        # layout: [fr*1, fr*2, fr*3, FTF*1, FTF*2, FTF*3, BPFO*1, ..., BPFI*1, ...]
        # fundamental of each type is at stride 3: indices 0,3,6,9,12
        fr, ftf, bpfo, bsf2, bpfi = freqs[0], freqs[3], freqs[6], freqs[9], freqs[12]
        print(f"  {cond:<20} {fr:7.2f} {ftf:7.2f} {bpfo:8.2f} {bsf2:8.2f} {bpfi:8.2f}")

    # ── window count (non-overlapping, win=4096) ──────────────────────────────
    print("\nDataset size (win=4096, stride=4096):")
    WIN = 4096
    label_counts = defaultdict(int)
    from scipy.io import loadmat as _loadmat
    sample_file  = all_mat[0]
    cond, bear, _ = parse_fname(sample_file)
    key = os.path.basename(sample_file).replace(".mat", "")
    mat = _loadmat(sample_file, struct_as_record=False, squeeze_me=True)
    vib_len = len(np.array(mat[key].X[1].Data).squeeze())
    wins_per_file = (vib_len - WIN) // WIN + 1
    print(f"  Signal length per file: {vib_len:,} samples  "
          f"→ {wins_per_file} windows/file")

    for cond_key in sorted(COND_RPM):
        for bid in DAMAGE_CATEGORY:
            if bid not in LABEL_MAP:
                continue
            n_files = sum(1 for p in all_mat if
                          os.path.basename(p).startswith(f"{cond_key}_{bid}_"))
            label_counts[(cond_key, LABEL_NAMES[LABEL_MAP[bid]])] += (
                n_files * wins_per_file
            )

    from bearmamba3.data_pu import COND_TRAIN, COND_TEST
    for split_name, conds in [("TRAIN (N09+N15_M01)", COND_TRAIN),
                               ("TEST  (N15_M07_F10)", COND_TEST)]:
        total = sum(v for (c, _), v in label_counts.items() if c in conds)
        by_class = defaultdict(int)
        for (c, lname), v in label_counts.items():
            if c in conds:
                by_class[lname] += v
        print(f"  {split_name}: {total:,} windows  "
              f"{dict(sorted(by_class.items()))}")

    print("\nDone.")


if __name__ == "__main__":
    main()
