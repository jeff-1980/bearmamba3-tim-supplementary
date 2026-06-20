#!/bin/bash
# exp07: 1D-CNN & Transformer-1D baselines, CWRU 4-class, EAAI SNR grid
# Usage: tmux new -s exp07 && bash experiments/exp07_baselines/run_exp07.sh
set -e
source ~/论文8/venv/bin/activate
cd ~/论文8

TRAIN="experiments/exp01_cwru_baseline/train.py"
SNR_TAGS=(clean snr0 snr-2 snr-4 snr-6 snr-8)
BACKBONES=(cnn1d transformer1d)

total=0
for bb in "${BACKBONES[@]}"; do
    for snr in "${SNR_TAGS[@]}"; do
        cfg="experiments/exp07_baselines/config_${bb}_${snr}.yaml"
        for seed in 0 1 2 3 4; do
            echo "=== $bb | $snr | seed=$seed ==="
            python $TRAIN --config $cfg --seed $seed
            total=$((total+1))
        done
    done
done

echo "All done. Total runs: $total"
