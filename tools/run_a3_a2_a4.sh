#!/usr/bin/env bash
# tools/run_a3_a2_a4.sh
# A3 (BearMamba-2 SNR网格) → A2 (10类 ±L_kin) → A4 (λ扫描 @SNR=0)
# 幂等：每个 run 前检查 summary.json 是否已存在，存在则跳过。
set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

TRAIN="experiments/exp01_cwru_baseline/train.py"

run_if_needed() {
    local config="$1"
    local results_dir="$2"
    local label="$3"
    if [[ -f "${results_dir}/summary.json" ]]; then
        echo "  [SKIP] ${label} — summary.json already exists"
        python tools/summarize_ablation.py "${results_dir}/"
        return 0
    fi
    echo ""
    echo "============================================================"
    echo "  ${label}"
    echo "============================================================"
    python "$TRAIN" --config "$config"
    python tools/summarize_ablation.py "${results_dir}/"
}

echo "================================================================"
echo "  A3: BearMamba-2 同管道 SNR 网格 (EAAI 锚定 6点, CE-only)"
echo "================================================================"
for snr in -8 -6 -4 -2 0 10; do
    tag="snr${snr}"
    run_if_needed \
        "experiments/exp04_mamba2/config_${tag}.yaml" \
        "results/exp04_mamba2_${tag}" \
        "A3: Mamba-2 SNR=${snr}dB"
done

echo ""
echo "================================================================"
echo "  A3 完成 — 打印 Mamba-2 SNR 网格汇总"
echo "================================================================"
python tools/summarize_ablation.py \
    results/exp04_mamba2_snr-8 \
    results/exp04_mamba2_snr-6 \
    results/exp04_mamba2_snr-4 \
    results/exp04_mamba2_snr-2 \
    results/exp04_mamba2_snr0  \
    results/exp04_mamba2_snr10

echo ""
echo "================================================================"
echo "  A2: CWRU 10类 SNR=0 ±L_kin"
echo "================================================================"
run_if_needed \
    "experiments/exp03_10class/config_snr0_nokin.yaml" \
    "results/exp03_10class_snr0_nokin" \
    "A2: 10class CE-only SNR=0"
run_if_needed \
    "experiments/exp03_10class/config_snr0_kin.yaml" \
    "results/exp03_10class_snr0_kin" \
    "A2: 10class +L_kin SNR=0"

echo ""
echo "================================================================"
echo "  A4: λ 扫描 {1e-3,1e-2,1e-1,1.0} @ SNR=0dB"
echo "================================================================"
for lam_tag in lk1em3 lk1em2 lk1em1 lk1p0; do
    run_if_needed \
        "experiments/exp05_lambda_scan/config_${lam_tag}.yaml" \
        "results/exp05_${lam_tag}" \
        "A4: λ=${lam_tag}"
done

echo ""
echo "================================================================"
echo "  A4 汇总 — λ 扫描"
echo "================================================================"
python tools/summarize_ablation.py \
    results/exp05_lk1em3 \
    results/exp05_lk1em2 \
    results/exp05_lk1em1 \
    results/exp05_lk1p0

echo ""
echo "================================================================"
echo "  A3→A2→A4 队列全部完成。可执行 /checkpoint M3。"
echo "================================================================"
