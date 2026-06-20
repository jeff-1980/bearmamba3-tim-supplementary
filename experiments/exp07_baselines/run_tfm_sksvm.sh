#!/bin/bash
# Restart: Transformer1D (20 epochs) + SK-SVM only
# CNN1D results already saved.
set -e
source ~/论文8/venv/bin/activate
cd ~/论文8

TRAIN="experiments/exp01_cwru_baseline/train.py"
SNR_TAGS=(clean snr0 snr-2 snr-4 snr-6 snr-8)

for snr in "${SNR_TAGS[@]}"; do
    cfg="experiments/exp07_baselines/config_transformer1d_${snr}.yaml"
    for seed in 0 1 2 3 4; do
        echo ">>> transformer1d | $snr | seed=$seed"
        python $TRAIN --config $cfg --seed $seed
    done
done

echo ">>> SK-SVM (all SNR, 5 seeds)"
python experiments/exp07_baselines/train_sksvm.py --run_all

echo "=============================="
echo "Transformer + SK-SVM DONE"
echo "=============================="
