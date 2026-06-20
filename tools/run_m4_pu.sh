#!/usr/bin/env bash
# tools/run_m4_pu.sh
# M4 PU 跨工况矩阵：BM3+nokin / BM3+kin / BM2+nokin × 5 seeds
# 协议：cross-condition same-bearing，训练 N09+N15_M01 → 测试 N15_M07_F10
# 幂等：summary.json 已存在则跳过
# 预计总时长 ~12.5h（3 run × ~4.2h each）
set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

TRAIN="experiments/exp06_pu/train.py"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

run_if_needed() {
    local config="$1"
    local results_dir="$2"
    local label="$3"
    if [[ -f "${results_dir}/summary.json" ]]; then
        echo "  [SKIP] ${label} — summary.json already exists"
        return 0
    fi
    echo ""
    echo "============================================================"
    echo "  ${label}"
    echo "============================================================"
    python "$TRAIN" --config "$config"
}

echo "================================================================"
echo "  M4: PU 跨工况矩阵（D17 路径 Y，3 条件 × 5 seeds）"
echo "  协议: cross-condition, same-bearing (NOT cross-bearing)"
echo "  训练: N09_M07_F10 + N15_M01_F10  →  测试: N15_M07_F10"
echo "================================================================"

run_if_needed \
    "experiments/exp06_pu/config_nokin.yaml" \
    "results/exp06_pu_nokin" \
    "M4-1: BM3 CE-only"

run_if_needed \
    "experiments/exp06_pu/config_kin.yaml" \
    "results/exp06_pu_kin" \
    "M4-2: BM3 +L_kin"

run_if_needed \
    "experiments/exp06_pu/config_bm2_nokin.yaml" \
    "results/exp06_pu_bm2_nokin" \
    "M4-3: BM2 CE-only"

echo ""
echo "================================================================"
echo "  M4 全部完成。汇总如下:"
echo "================================================================"
python3 -c "
import json, glob, numpy as np
for exp, label in [
    ('exp06_pu_nokin',    'BM3 CE-only'),
    ('exp06_pu_kin',      'BM3 +L_kin '),
    ('exp06_pu_bm2_nokin','BM2 CE-only'),
]:
    files = sorted(glob.glob(f'results/{exp}/seed_*.json'))
    if not files:
        print(f'  {label}: NO DATA')
        continue
    accs = [json.load(open(f)).get('best_val_acc',0) for f in files]
    print(f'  {label}: {np.mean(accs)*100:.2f}±{np.std(accs,ddof=1)*100:.2f}%  {[f\"{a*100:.2f}\" for a in accs]}')
"

echo ""
echo "  可执行 /checkpoint M4"
echo "================================================================"
