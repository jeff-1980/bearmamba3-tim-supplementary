# BearMamba-3: Mamba-3 State-Space Bearing Fault Diagnosis with Kinematic Frequency Inductive Bias and Multi-Sensor Fusion

Supplementary code and experimental artifacts for the paper submitted to
**IEEE Transactions on Instrumentation and Measurement (TIM)**.

---

## Repository contents

```
.
├── bearmamba3/           Core model: Mamba-3 backbone + kinematic loss
├── baselines/            Baseline implementations (Mamba-2, 1D-CNN ± BN, Transformer-1D)
├── experiments/          Training scripts and YAML configs for every experiment
├── tools/                Data inventory and visualization utilities
├── results/
│   ├── figures/          Final paper figures (PDF + PNG)
│   ├── tsne_feats/       Multi-seed t-SNE feature caches
│   └── exp*/summary.json Aggregated 5-seed results for every experiment
├── paper/                LaTeX source, references.bib, compiled main.pdf
├── setup_env.sh          Environment setup script
├── protocol_diff.md      Protocol differences vs. prior published work
├── requirements.txt      Python dependencies
└── README.md             This file
```

**Not included** (available separately upon acceptance via IEEE DataPort):
- Raw per-step `.npz` snapshots (1085 files, ~1.7 GB) — `I1` frequency
  snapshots and per-epoch state distributions
- Trained model checkpoints (none stored on disk; all results are re-runnable)
- Raw vibration data (public benchmarks; see *Datasets* below)

---

## Datasets (public)

All training/evaluation uses three **public** bearing benchmarks. Download
links and bearing-geometry parameters used in the paper:

| Dataset | URL | Bearings used | Sampling rate |
|---|---|---|---|
| **CWRU** | https://engineering.case.edu/bearingdatacenter | SKF 6205 (DE) + 6203 (FE) | 12 kHz |
| **Paderborn (PU)** | https://mb.uni-paderborn.de/kat/forschung/kat-datacenter | SKF 6203 | 64 kHz |
| **XJTU-SY** | https://biaowang.tech/xjtu-sy-bearing-datasets | LDK UER204 | 25.6 kHz |

Place the unpacked datasets under `~/data/{cwru,pu,xjtu}/` or adjust
`config.yaml` paths in each experiment folder accordingly.

---

## Reproducibility checklist

| Element | Status |
|---|---|
| Five independent random seeds {0,1,2,3,4} | ✅ All experiments |
| Mean ± std (sample stdev, ddof=1) | ✅ |
| Wilcoxon signed-rank statistical test | ✅ |
| Fixed hyperparameters across datasets | ✅ See `experiments/*/config*.yaml` |
| Negative-result reporting | ✅ See paper Discussion §5 |
| Honest scope statement (1D-CNN within-condition advantage) | ✅ See paper Discussion §5.6 |

---

## Quick start

### 1. Environment

```bash
# Linux (Ubuntu 22.04+ recommended), CUDA 12.x, Python 3.10
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install official Mamba-3 (required for L_kin)
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir \
    git+https://github.com/state-spaces/mamba.git --no-build-isolation
```

### 2. Hardware

- **Minimum**: NVIDIA GPU with ≥ 4 GB VRAM (training Mamba-3 SISO)
- **Recommended**: RTX A5000 / A100 (used in our experiments)
- **MIMO mode** (`is_mimo=True`) requires sm_90+ (Hopper) — see paper §3.3
  for the sm_86 limitation analysis

### 3. Run an experiment

Each experiment under `experiments/expXX_*/` ships with `train.py` and one
or more `config*.yaml` files. Example: reproduce the headline CWRU
SNR-ablation curve (C2 main figure):

```bash
cd experiments/exp02_snr_ablation
# Train all 5 seeds at SNR = -4 dB with L_kin
for seed in 0 1 2 3 4; do
    python3 train.py --config config_kin.yaml --snr -4 --seed $seed
done
# Aggregate
python3 aggregate.py results/exp02_snr-4_kin/
```

### 4. Reproduce paper figures from cached summaries

The `results/exp*/summary.json` files contain the per-seed numerical
results that back every table and figure in the paper. The Step 5
visualization scripts under `experiments/step5_*.py` regenerate the
figures from these summaries plus the cached t-SNE features in
`results/tsne_feats/`:

```bash
python3 experiments/step5_i1_freqdist.py    # Fig. 6 (CWRU I1)
python3 experiments/step5_xjtu_i1.py        # Fig. 7 (XJTU I1)
python3 experiments/step5_tsne.py           # Fig. 8 (single-seed t-SNE)
python3 experiments/step5_tsne_multiseed.py # Fig. 9 (5-seed t-SNE composite)
python3 experiments/step5_b2_snr_curve.py   # Fig. 10 (CWRU dual-sensor)
python3 experiments/step5_snr_curve_paradigm.py  # Fig. 11 (BN vs. L_kin paradigm)
```

---

## Three-way correspondence with paper claims

| Paper claim | Code path | Result file |
|---|---|---|
| **C1**: Mamba-3 first adoption for bearing diagnosis | `bearmamba3/model.py`, `experiments/exp01_cwru_baseline/`, `exp06_pu/` | `results/exp01_cwru_*/summary.json`, `results/exp06_pu_*/summary.json` |
| **C2**: Kinematic-frequency inductive bias L_kin | `bearmamba3/kinematic_loss.py`, `experiments/exp02_snr_ablation/`, `step5_i1_*.py` | `results/exp02_snr*/summary.json`, `results/figures/i1_*.pdf` |
| **C3**: Multi-sensor precision–sample-efficiency trade-off | `experiments/exp_b2_dual_sensor/`, `exp_xjtu/` | `results/b2_snr_curve_summary.json`, `results/b4_5_summary.json` |
| **M_ext Phase 1/2/2b**: 1D-CNN cross-dataset analysis (D33) | `experiments/exp_mext_e1*/`, `exp_mext_e2*/`, `exp_mext_e21b_*/` | `results/exp_mext_*/summary.json` |

---

## Key methodological documents

| File | Content |
|---|---|
| `protocol_diff.md` | Window length, noise injection, fault-frequency conventions vs. prior published work |
| `paper/journal_switch_notes.md` | LaTeX template migration notes (Elsevier → IEEEtran) |
| `paper/pgtmt_crossspeed_check.md` | Concurrent-work positioning analysis |
| `paper/references_verification.md` | Reference list verification report (54 entries) |

---

## Citation

```bibtex
@article{wang2026bearmamba3,
  author  = {Wang, Yi and Tang, Yongqing},
  title   = {{BearMamba-3}: {Mamba-3} State-Space Bearing Fault Diagnosis
             with Kinematic Frequency Inductive Bias and Multi-Sensor Fusion},
  journal = {IEEE Transactions on Instrumentation and Measurement},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT License. See `LICENSE` file.

---

## Contact

For questions about reproducibility:
- **Corresponding author**: Yi Wang (`jefflsxy@lsu.edu.cn`)
- **Co-author**: Yongqing Tang (`yq_tang@lyun.edu.cn`)
