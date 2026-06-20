#!/usr/bin/env bash
# tools/run_b1_pu_noise.sh
# B1: PU+noise SNR 扫描（BM3 ±L_kin, SNR=0/-4dB, 5seeds）
# 目的：补充 C2 变速+噪声叙事支撑（M4 为干净信号，L_kin 效果不显著）
# 幂等：summary.json 已存在则跳过
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
echo "  B1: PU+noise SNR 扫描（C2 变速+噪声叙事支撑）"
echo "  协议: cross-condition, same-bearing，BM3 ±L_kin × 2 SNR × 5 seeds"
echo "================================================================"

run_if_needed \
    "experiments/exp06_pu/config_nokin_snr0.yaml" \
    "results/exp06_pu_nokin_snr0" \
    "B1-1: BM3 CE-only @ SNR=0dB"

run_if_needed \
    "experiments/exp06_pu/config_kin_snr0.yaml" \
    "results/exp06_pu_kin_snr0" \
    "B1-2: BM3 +L_kin @ SNR=0dB"

run_if_needed \
    "experiments/exp06_pu/config_nokin_snrm4.yaml" \
    "results/exp06_pu_nokin_snrm4" \
    "B1-3: BM3 CE-only @ SNR=-4dB"

run_if_needed \
    "experiments/exp06_pu/config_kin_snrm4.yaml" \
    "results/exp06_pu_kin_snrm4" \
    "B1-4: BM3 +L_kin @ SNR=-4dB"

echo ""
echo "================================================================"
echo "  B1 全部完成。汇总如下:"
echo "================================================================"
python3 -c "
import json, glob, numpy as np
from scipy.stats import wilcoxon

results = {}
for exp, label, snr in [
    ('exp06_pu_nokin_snr0',  'BM3 CE-only', '0dB'),
    ('exp06_pu_kin_snr0',    'BM3 +L_kin ', '0dB'),
    ('exp06_pu_nokin_snrm4', 'BM3 CE-only', '-4dB'),
    ('exp06_pu_kin_snrm4',   'BM3 +L_kin ', '-4dB'),
]:
    files = sorted(glob.glob(f'results/{exp}/seed_*.json'))
    if not files:
        print(f'  SNR={snr} {label}: NO DATA')
        continue
    accs = [json.load(open(f)).get('best_val_acc', 0) for f in files]
    results[(snr, label.strip())] = accs
    mean, std = np.mean(accs)*100, np.std(accs, ddof=1)*100
    print(f'  SNR={snr:4s} {label}: {mean:.2f}±{std:.2f}%  {[f\"{a*100:.2f}\" for a in accs]}')

print()
for snr in ['0dB', '-4dB']:
    nokin = results.get((snr, 'BM3 CE-only'))
    kin   = results.get((snr, 'BM3 +L_kin'))
    if nokin and kin:
        d = np.mean(np.array(kin) - np.array(nokin)) * 100
        _, p = wilcoxon(kin, nokin)
        print(f'  SNR={snr}: L_kin effect = {d:+.2f}pp  Wilcoxon p={p:.4f}')
"

echo ""
echo "  可执行 /checkpoint B1"
echo "================================================================"
