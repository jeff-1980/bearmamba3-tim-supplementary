#!/usr/bin/env python3
"""
BLOCK-B4-1 fix: recompute f_fault in all existing XJTU L_kin snapshots.

All test snapshots were taken from Cond3 (40Hz10kN = 2400rpm) bearings.
The original code called compute_fault_freqs() without bearing_kwargs, so it
used CWRU SKF 6205 geometry instead of XJTU LDK UER204. This script loads
each .npz, recomputes f_fault with correct params at rpm=2400, and overwrites.

XJTU UER204: Z=8, d=7.92mm, D=34.55mm, alpha=0°
Cond3 nominal: 2400rpm → fr=40Hz
Expected:  BPFO=123.32Hz  BPFI=196.68Hz  (vs wrong BPFO=143.41Hz)
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))  # ~/论文8

import numpy as np
import torch

from bearmamba3.kinematic_loss import compute_fault_freqs

XJTU_BEARING_KWARGS = dict(n_balls=8, d=7.92, D=34.55)
COND3_RPM = 2400.0

SNAP_DIRS = [
    pathlib.Path("/home/jeffwork/论文8/results/exp_xjtu_lobo_kin"),
    pathlib.Path("/home/jeffwork/论文8/results/exp_xjtu_cross_kin"),
]

device = torch.device("cpu")


def recompute_f_fault(n_samples: int) -> np.ndarray:
    rpm = torch.full((n_samples,), COND3_RPM, dtype=torch.float32, device=device)
    f_fault = compute_fault_freqs(rpm, device=device, **XJTU_BEARING_KWARGS)
    return f_fault.cpu().numpy().astype(np.float32)


def fix_snapshot(path: pathlib.Path, dry_run: bool = False) -> dict:
    data = dict(np.load(path))
    n = data["f_bar"].shape[0]

    # column 6 = BPFO (fr, 2fr, 3fr, FTF, 2FTF, 3FTF, BPFO, ...)
    bpfo_col = 6
    old_bpfo = float(data["f_fault"][0, bpfo_col])
    new_f_fault = recompute_f_fault(n)
    new_bpfo = float(new_f_fault[0, bpfo_col])

    if not dry_run:
        data["f_fault"] = new_f_fault
        np.savez_compressed(path, **data)

    return {"path": path.name, "old_bpfo_sample0": old_bpfo, "new_bpfo_sample0": new_bpfo}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print changes without writing")
    args = parser.parse_args()

    total = 0
    errors = 0
    for d in SNAP_DIRS:
        if not d.exists():
            print(f"[SKIP] {d} does not exist")
            continue
        npz_files = sorted(d.glob("*_kin_ep*.npz"))
        print(f"\n{d.name}: {len(npz_files)} snapshots")
        for p in npz_files:
            try:
                info = fix_snapshot(p, dry_run=args.dry_run)
                tag = "[DRY]" if args.dry_run else "[OK ]"
                print(f"  {tag} {info['path'][:70]:70s}  BPFO: {info['old_bpfo_sample0']:7.2f} → {info['new_bpfo_sample0']:7.2f} Hz")
                total += 1
            except Exception as e:
                print(f"  [ERR] {p.name}: {e}")
                errors += 1

    print(f"\n{'[DRY-RUN] ' if args.dry_run else ''}Fixed {total} files, {errors} errors.")

    # Verification: show expected values
    if not args.dry_run:
        print("\nVerification (sample from first found snapshot):")
        for d in SNAP_DIRS:
            sample = next(d.glob("*_kin_ep*.npz"), None)
            if sample:
                data = np.load(sample)
                print(f"  {sample.name[:60]}")
                print(f"    f_fault[0,:8] = {data['f_fault'][0,:8]}")
                print(f"    (expect fr≈40, BPFO≈123.32, BPFI≈196.68)")
                break
