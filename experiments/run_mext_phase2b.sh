#!/bin/bash
# M_ext Phase 2b: 1D-CNN-no-BN CWRU SNR 扫描（BLOCK-P2-1 Option A）
# 6 SNR × 5 seeds = 30 runs
# 目的：验证 BN 是否是 1D-CNN 在 CWRU 低 SNR 领先的根因

set -e
source ~/论文8/venv/bin/activate
cd ~/论文8

T0=$(date +%s)
log() { echo "[$(date '+%H:%M:%S')] $*"; }

TRAIN="experiments/exp01_cwru_baseline/train.py"

log "=== Phase 2b: 1D-CNN-no-BN CWRU SNR scan (6 SNR × 5 seeds) ==="
for snr_label in snr-8 snr-6 snr-4 snr-2 snr0 clean; do
  cfg="experiments/exp_mext_e21b_1dcnn_nobn_cwru/config_${snr_label}.yaml"
  log "  ${snr_label} ..."
  python $TRAIN --config $cfg
  python3 -c "
import json, glob, numpy as np
files = sorted(glob.glob('results/exp_mext_e21b_1dcnn_nobn_cwru_${snr_label}/seed_*.json'))
accs  = [json.load(open(f))['best_val_acc']*100 for f in files]
print(f'    ${snr_label}: {np.mean(accs):.2f}±{np.std(accs,ddof=1):.2f}%  n={len(accs)}')
"
done

T1=$(date +%s)
log "=============================="
log "Phase 2b COMPLETE  elapsed=$(( (T1-T0)/60 ))min"
log "=============================="

# Quick summary vs 1D-CNN-with-BN
python3 - << 'PYEOF'
import json, glob, numpy as np

print("\n=== Phase 2b vs 1D-CNN (with BN) comparison ===")
print(f"{'SNR':>6}  {'NoBN mean±std':>18}  {'BN mean±std':>18}  {'Δ(NoBN-BN)':>12}")

snr_map = {
    'snr-8': (-8, 'exp07_cnn1d_snr-8'),
    'snr-6': (-6, 'exp07_cnn1d_snr-6'),
    'snr-4': (-4, 'exp07_cnn1d_snr-4'),
    'snr-2': (-2, 'exp07_cnn1d_snr-2'),
    'snr0':  ( 0, 'exp07_cnn1d_snr0'),
    'clean': (10, 'exp07_cnn1d_clean'),
}

for label, (snr_db, bn_dir) in snr_map.items():
    nobn_files = sorted(glob.glob(f'results/exp_mext_e21b_1dcnn_nobn_cwru_{label}/seed_*.json'))
    bn_files   = sorted(glob.glob(f'results/{bn_dir}/seed_*.json'))
    if not nobn_files or not bn_files:
        continue
    nobn = np.array([json.load(open(f))['best_val_acc']*100 for f in nobn_files])
    bn   = np.array([json.load(open(f))['best_val_acc']*100 for f in bn_files])
    delta = np.mean(nobn) - np.mean(bn)
    print(f"{snr_db:>6}dB  {np.mean(nobn):6.2f}±{np.std(nobn,ddof=1):.2f}%  "
          f"{np.mean(bn):6.2f}±{np.std(bn,ddof=1):.2f}%  "
          f"{delta:+.2f}pp")
PYEOF
