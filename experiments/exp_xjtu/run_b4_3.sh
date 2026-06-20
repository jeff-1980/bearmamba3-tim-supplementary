#!/bin/bash
# experiments/exp_xjtu/run_b4_3.sh
# B4-3: XJTU-SY BM3 ±L_kin × LOBO + 跨工况 (D26)
#
# LOBO (Cond3, 4 folds × 5 seeds × 2 configs):  ~13h
# Cross (Cond2→Cond3, 5 seeds × 2 configs):      ~6h
# 总计:  ~19h，幂等（summary.json 存在则跳过）
#
# 用法:
#   tmux new -s b4_3
#   source venv/bin/activate
#   bash experiments/exp_xjtu/run_b4_3.sh 2>&1 | tee results/b4_3.log

set -e
cd /home/jeffwork/论文8
source venv/bin/activate

TRAIN="experiments/exp_xjtu/train.py"

run_if_needed() {
    local cfg="$1"
    local results_dir="$2"
    if [ -f "${results_dir}/summary.json" ]; then
        echo "[SKIP] ${results_dir} already complete"
        return
    fi
    echo ""
    echo "========================================================"
    echo "[RUN ] $cfg"
    echo "========================================================"
    python "$TRAIN" --config "$cfg"
}

echo "========================================================"
echo "  B4-3 XJTU-SY Training Queue (D26)"
echo "  $(date)"
echo "========================================================"

# ── LOBO ──────────────────────────────────────────────────────────
run_if_needed \
    experiments/exp_xjtu/config_lobo_nokin.yaml \
    ~/论文8/results/exp_xjtu_lobo_nokin

run_if_needed \
    experiments/exp_xjtu/config_lobo_kin.yaml \
    ~/论文8/results/exp_xjtu_lobo_kin

# ── Cross-condition ────────────────────────────────────────────────
run_if_needed \
    experiments/exp_xjtu/config_cross_nokin.yaml \
    ~/论文8/results/exp_xjtu_cross_nokin

run_if_needed \
    experiments/exp_xjtu/config_cross_kin.yaml \
    ~/论文8/results/exp_xjtu_cross_kin

echo ""
echo "========================================================"
echo "  B4-3 ALL DONE  $(date)"
echo "========================================================"

# ── Quick summary ──────────────────────────────────────────────────
python3 -c "
import json, pathlib, numpy as np

ROOT = pathlib.Path('results')
configs = [
    ('exp_xjtu_lobo_nokin',  'LOBO   CE-only'),
    ('exp_xjtu_lobo_kin',    'LOBO   +L_kin '),
    ('exp_xjtu_cross_nokin', 'Cross  CE-only'),
    ('exp_xjtu_cross_kin',   'Cross  +L_kin '),
]
print()
print(f'{'Config':22s}  metric         mean±std    per-seed')
for name, label in configs:
    p = ROOT / name / 'summary.json'
    if not p.exists():
        print(f'{label:22s}  [MISSING]')
        continue
    d = json.load(open(p))
    if d['mode'] == 'lobo':
        m = d['mean_macro_recall'] * 100
        s = d['std_macro_recall'] * 100
        vals = [f'{v*100:.2f}' for v in d['per_seed_macro_recall']]
        print(f'{label:22s}  macro_recall   {m:.2f}±{s:.2f}%  [{chr(47).join(vals)}]')
    else:
        m = d['mean_macro_f1'] * 100
        s = d['std_macro_f1'] * 100
        vals = [f'{v*100:.2f}' for v in d['macro_f1s']]
        print(f'{label:22s}  macro_F1       {m:.2f}±{s:.2f}%  [{chr(47).join(vals)}]')
"
