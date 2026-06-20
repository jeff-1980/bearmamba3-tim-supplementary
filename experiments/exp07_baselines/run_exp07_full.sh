#!/bin/bash
# exp07: CNN1D + Transformer1D + SK-SVM, CWRU 4-class, 全 EAAI SNR 网格
# tmux: tmux new -s exp07 && bash experiments/exp07_baselines/run_exp07_full.sh
set -e
source ~/论文8/venv/bin/activate
cd ~/论文8

TRAIN="experiments/exp01_cwru_baseline/train.py"
SNR_TAGS=(clean snr0 snr-2 snr-4 snr-6 snr-8)
DEEP_BACKBONES=(cnn1d transformer1d)

# ── 深度学习基线（CNN + Transformer）──────────────────────────────
for bb in "${DEEP_BACKBONES[@]}"; do
    for snr in "${SNR_TAGS[@]}"; do
        cfg="experiments/exp07_baselines/config_${bb}_${snr}.yaml"
        for seed in 0 1 2 3 4; do
            echo ">>> $bb | $snr | seed=$seed"
            python $TRAIN --config $cfg --seed $seed
        done
    done
done

# ── SK-SVM 基线（全 SNR 网格, 5 seeds）────────────────────────────
echo ">>> SK-SVM (all SNR, 5 seeds)"
python experiments/exp07_baselines/train_sksvm.py --run_all

echo "=============================="
echo "exp07 ALL DONE"
echo "=============================="
