#!/bin/bash
# M_ext Phase 2: BM3+BN vs 1D-CNN fair comparison
# E2.1: BM3+BN CE-only CWRU 6 SNR × 5 seeds (30 runs)
# E2.2: BM3+BN +L_kin  CWRU 6 SNR × 5 seeds (30 runs)
# E2.3: BM3+BN CE-only XJTU cross (5 seeds, OOD Win1 closure)
# Total ~60 runs; estimated GPU time ~5h

set -e
source ~/论文8/venv/bin/activate
cd ~/论文8

T0=$(date +%s)
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── E2.1: BM3+BN CE-only CWRU SNR 扫描 ───────────────────────────
log "=== E2.1: BM3+BN CE-only CWRU SNR scan (6 SNR × 5 seeds) ==="
TRAIN="experiments/exp01_cwru_baseline/train.py"
for snr_label in snr-8 snr-6 snr-4 snr-2 snr0 clean; do
  cfg="experiments/exp_mext_e21_bm3bn_cwru/config_${snr_label}.yaml"
  log "  E2.1 ${snr_label} ..."
  python $TRAIN --config $cfg
  python3 -c "
import json, glob, numpy as np
files=sorted(glob.glob('results/exp_mext_e21_bm3bn_cwru_${snr_label}/seed_*.json'))
accs=[json.load(open(f))['best_val_acc']*100 for f in files]
print(f'    ${snr_label}: {np.mean(accs):.2f}±{np.std(accs,ddof=1):.2f}%  n={len(accs)}')
"
done
log "E2.1 DONE"

# ── E2.2: BM3+BN +L_kin CWRU SNR 扫描 ───────────────────────────
log "=== E2.2: BM3+BN +L_kin CWRU SNR scan (6 SNR × 5 seeds) ==="
for snr_label in snr-8 snr-6 snr-4 snr-2 snr0 clean; do
  cfg="experiments/exp_mext_e22_bm3bn_kin_cwru/config_${snr_label}.yaml"
  log "  E2.2 ${snr_label} ..."
  python $TRAIN --config $cfg
  python3 -c "
import json, glob, numpy as np
files=sorted(glob.glob('results/exp_mext_e22_bm3bn_kin_cwru_${snr_label}/seed_*.json'))
accs=[json.load(open(f))['best_val_acc']*100 for f in files]
print(f'    ${snr_label}: {np.mean(accs):.2f}±{np.std(accs,ddof=1):.2f}%  n={len(accs)}')
"
done
log "E2.2 DONE"

# ── E2.3: BM3+BN XJTU 跨工况 ─────────────────────────────────────
log "=== E2.3: BM3+BN XJTU cross Cond2→Cond3 (5 seeds) ==="
python experiments/exp_xjtu/train.py \
  --config experiments/exp_mext_e23_bm3bn_xjtu_cross/config.yaml
log "E2.3 DONE → results/exp_mext_e23_bm3bn_xjtu_cross/summary.json"
python3 -c "
import json
d=json.load(open('results/exp_mext_e23_bm3bn_xjtu_cross/summary.json'))
print(f'  E2.3 BM3+BN XJTU cross: macro_F1={d[\"mean_macro_f1\"]*100:.2f}±{d[\"std_macro_f1\"]*100:.2f}%')
print(f'  per_seed: {[round(x*100,2) for x in d[\"macro_f1s\"]]}')
"

T1=$(date +%s)
log "=============================="
log "Phase 2 COMPLETE  elapsed=$(( (T1-T0)/60 ))min"
log "=============================="
