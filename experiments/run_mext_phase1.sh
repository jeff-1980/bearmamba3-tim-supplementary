#!/bin/bash
# M_ext Phase 1: 1D-CNN cross-dataset pressure test
# E1.1 PU cross (5 seeds) → E1.2 XJTU LOBO (4folds×5seeds) →
# E1.3 XJTU cross (5 seeds) → E1.4 CWRU dual SNR×5 (25 seeds)
# Total ~55 runs; estimated GPU time 5-6h

set -e
source ~/论文8/venv/bin/activate
cd ~/论文8

T0=$(date +%s)
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── E1.1: 1D-CNN PU 跨工况 ────────────────────────────────────────
log "=== E1.1: 1D-CNN PU cross-condition (5 seeds) ==="
python experiments/exp06_pu/train.py \
  --config experiments/exp_mext_e11_1dcnn_pu_cross/config.yaml
log "E1.1 DONE → results/exp_mext_e11_1dcnn_pu_cross/summary.json"
python3 -c "
import json
d=json.load(open('results/exp_mext_e11_1dcnn_pu_cross/summary.json'))
print(f'  E1.1 PU cross 1D-CNN: {d[\"mean_best_val_acc\"]*100:.2f}±{d[\"std_best_val_acc\"]*100:.2f}%')
print(f'  per_seed: {[round(x*100,2) for x in d[\"best_val_accs\"]]}')
"

# ── E1.2: 1D-CNN XJTU LOBO ──────────────────────────────────────
log "=== E1.2: 1D-CNN XJTU LOBO Cond3 (4folds × 5 seeds) ==="
python experiments/exp_xjtu/train.py \
  --config experiments/exp_mext_e12_1dcnn_xjtu_lobo/config.yaml
log "E1.2 DONE → results/exp_mext_e12_1dcnn_xjtu_lobo/summary.json"
python3 -c "
import json
d=json.load(open('results/exp_mext_e12_1dcnn_xjtu_lobo/summary.json'))
print(f'  E1.2 XJTU LOBO 1D-CNN: macro_recall={d[\"mean_macro_recall\"]*100:.2f}±{d[\"std_macro_recall\"]*100:.2f}%')
print(f'  per_seed: {[round(x*100,2) for x in d[\"per_seed_macro_recall\"]]}')
"

# ── E1.3: 1D-CNN XJTU 跨工况 ────────────────────────────────────
log "=== E1.3: 1D-CNN XJTU cross Cond2→Cond3 (5 seeds) ==="
python experiments/exp_xjtu/train.py \
  --config experiments/exp_mext_e13_1dcnn_xjtu_cross/config.yaml
log "E1.3 DONE → results/exp_mext_e13_1dcnn_xjtu_cross/summary.json"
python3 -c "
import json
d=json.load(open('results/exp_mext_e13_1dcnn_xjtu_cross/summary.json'))
print(f'  E1.3 XJTU cross 1D-CNN: macro_F1={d[\"mean_macro_f1\"]*100:.2f}±{d[\"std_macro_f1\"]*100:.2f}%')
print(f'  per_seed: {[round(x*100,2) for x in d[\"macro_f1s\"]]}')
"

# ── E1.4: 1D-CNN CWRU 双传感器 SNR 扫描 ─────────────────────────
log "=== E1.4: 1D-CNN CWRU dual-sensor SNR scan (5 SNR × 5 seeds) ==="
TRAIN="experiments/exp01_cwru_baseline/train.py"
for snr in -8 -6 -4 -2 0; do
  cfg="experiments/exp_mext_e14_1dcnn_cwru_dual/config_snr${snr}.yaml"
  log "  E1.4 SNR=${snr}dB ..."
  python $TRAIN --config $cfg
  python3 -c "
import json, glob, numpy as np
files=sorted(glob.glob('results/exp_mext_e14_1dcnn_cwru_dual_snr${snr}/seed_*.json'))
accs=[json.load(open(f))['best_val_acc']*100 for f in files]
print(f'    SNR=${snr}dB: {np.mean(accs):.2f}±{np.std(accs,ddof=1):.2f}%  n={len(accs)}')
"
done
log "E1.4 DONE"

T1=$(date +%s)
log "=============================="
log "Phase 1 COMPLETE  elapsed=$(( (T1-T0)/60 ))min"
log "=============================="
