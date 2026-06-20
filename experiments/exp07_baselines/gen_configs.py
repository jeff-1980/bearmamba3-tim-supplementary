"""
Generate config files for exp07_baselines.
Run once: python experiments/exp07_baselines/gen_configs.py
"""
import os
from pathlib import Path

BASEDIR = Path(__file__).parent
SNR_GRID = [None, 0, -2, -4, -6, -8]   # None = clean (no noise)
BACKBONES = ["cnn1d", "transformer1d"]

TEMPLATE = """\
name: {name}
data_dir: ~/论文8/data/cwru_12k_de
channels: [DE]
win_len: 2048
stride: 1024
val_ratio: 0.2
batch_size: 64
num_workers: 4
backbone: {backbone}
d_model: 64
n_layers: 4
n_classes: 4
conv_stride: 2
nhead: 4
dim_feedforward: 256
epochs: 50
lr: 3.0e-4
weight_decay: 1.0e-4
grad_clip: 1.0
scheduler: cosine
{snr_line}lambda_kin: 0.0
seeds: [0, 1, 2, 3, 4]
results_dir: ~/论文8/results/{name}
"""

for bb in BACKBONES:
    for snr in SNR_GRID:
        snr_tag = "clean" if snr is None else f"snr{snr:+d}".replace("+", "")
        name = f"exp07_{bb}_{snr_tag}"
        snr_line = "" if snr is None else f"noise_snr_db: {float(snr)}\n"
        cfg = TEMPLATE.format(name=name, backbone=bb, snr_line=snr_line)
        outpath = BASEDIR / f"config_{bb}_{snr_tag}.yaml"
        outpath.write_text(cfg)
        print(f"  wrote {outpath.name}")

print("Done.")
