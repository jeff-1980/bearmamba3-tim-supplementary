#!/bin/bash
# experiments/exp_b2_dual_sensor/run_b2.sh
# B2: CWRU DE+FE dual-sensor vs single-sensor × ±L_kin, SNR=0dB, 5 seeds
#
# 单传感器基准已完成（A1）：
#   CE-only:   results/exp02_snr0_nokin/ → 99.75±0.22%
#   +L_kin:    results/exp02_snr0_kin/  → 99.81±0.13%
#
# 本脚本只跑双传感器两条（nokin + kin），每条 5 seeds × 50 epochs
# 用法: bash experiments/exp_b2_dual_sensor/run_b2.sh [--smoke]

set -e
cd /home/jeffwork/论文8
source venv/bin/activate

TRAIN="experiments/exp01_cwru_baseline/train.py"  # generic train.py
SMOKE="${1:-}"  # pass --smoke for quick sanity check

run_if_needed() {
    local cfg="$1"
    local results_dir="$2"
    if [ -f "${results_dir}/summary.json" ]; then
        echo "[SKIP] ${results_dir} already complete"
        return
    fi
    echo "[RUN ] python $TRAIN --config $cfg $SMOKE"
    python "$TRAIN" --config "$cfg" $SMOKE
}

# B2-1: Dual-sensor CE-only
run_if_needed \
    experiments/exp_b2_dual_sensor/config_dual_nokin.yaml \
    ~/论文8/results/exp_b2_dual_nokin

# B2-2: Dual-sensor +L_kin
run_if_needed \
    experiments/exp_b2_dual_sensor/config_dual_kin.yaml \
    ~/论文8/results/exp_b2_dual_kin

# ── Summary ───────────────────────────────────────────────────────
if [ -z "$SMOKE" ]; then
    echo ""
    echo "============================================"
    echo "B2 Summary (single vs dual sensor, SNR=0dB)"
    echo "============================================"
    python3 -c "
import json, numpy as np

def load(p):
    with open(p) as f: return json.load(f)

single_nokin = load('results/exp02_snr0_nokin/summary.json')
single_kin   = load('results/exp02_snr0_kin/summary.json')
dual_nokin   = load('results/exp_b2_dual_nokin/summary.json')
dual_kin     = load('results/exp_b2_dual_kin/summary.json')

rows = [
    ('Single (DE)',    'CE-only', single_nokin),
    ('Single (DE)',    '+L_kin',  single_kin),
    ('Dual (DE+FE)',   'CE-only', dual_nokin),
    ('Dual (DE+FE)',   '+L_kin',  dual_kin),
]
print(f'{'Sensor':15s} {'Loss':8s} mean±std      seeds')
for name, loss, d in rows:
    accs = d['best_val_accs']
    m = np.mean(accs)*100
    s = np.std(accs, ddof=1)*100
    seed_str = '/'.join(f'{a*100:.2f}' for a in accs)
    print(f'{name:15s} {loss:8s} {m:.2f}±{s:.2f}%  [{seed_str}]')
"
fi
