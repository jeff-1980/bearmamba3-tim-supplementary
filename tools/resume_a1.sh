#!/usr/bin/env bash
# tools/resume_a1.sh
# Resume A1 EAAI grid from interruption:
#   - snr0_kin: seed 0 already done → run seeds 1-4, then merge summary
#   - snr10_nokin/kin: run all 5 seeds
set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

TRAIN="experiments/exp01_cwru_baseline/train.py"
EXP_DIR="experiments/exp02_snr_ablation"

echo "=== [1/3] SNR=0dB +L_kin — seeds 1-4 (seed 0 already saved) ==="
python "$TRAIN" --config "${EXP_DIR}/config_snr0_kin.yaml" --seeds 1 2 3 4

echo ""
echo "--- Merging snr0_kin summary from all 5 seed files ---"
python - <<'PY'
import json, numpy as np
from pathlib import Path

d = Path("results/exp02_snr0_kin")
files = sorted(d.glob("seed_*.json"))
accs = [json.loads(f.read_text())["best_val_acc"] for f in files]
cfg  = json.loads(files[0].read_text())   # borrow config fields from seed_0

summary = {
    "config": "config_snr0_kin.yaml",
    "backbone": "mamba3",
    "lambda_kin": 0.01,
    "noise_snr_db": 0.0,
    "label_mode": "4class",
    "seeds": list(range(len(accs))),
    "best_val_accs": accs,
    "mean_best_val_acc": float(np.mean(accs)),
    "std_best_val_acc": float(np.std(accs, ddof=1)),
}
out = d / "summary.json"
out.write_text(json.dumps(summary, indent=2))
print(f"  seeds: {[f'{a*100:.2f}%' for a in accs]}")
print(f"  mean±std: {np.mean(accs)*100:.2f}±{np.std(accs,ddof=1)*100:.2f}%")
print(f"  → {out}")
PY

echo ""
echo "=== [2/3] SNR=+10dB  CE-only ==="
python "$TRAIN" --config "${EXP_DIR}/config_snr10_nokin.yaml"
python tools/summarize_ablation.py results/exp02_snr10_nokin/

echo ""
echo "=== [3/3] SNR=+10dB  +L_kin (λ=0.01) ==="
python "$TRAIN" --config "${EXP_DIR}/config_snr10_kin.yaml"
python tools/summarize_ablation.py results/exp02_snr10_kin/

echo ""
echo "=== A1 EAAI grid COMPLETE — full table ==="
python tools/summarize_ablation.py \
    results/exp02_snr-8_nokin results/exp02_snr-8_kin \
    results/exp02_snr-6_nokin results/exp02_snr-6_kin \
    results/exp02_snr-4_nokin results/exp02_snr-4_kin \
    results/exp02_snr-2_nokin results/exp02_snr-2_kin \
    results/exp02_snr0_nokin  results/exp02_snr0_kin  \
    results/exp02_snr10_nokin results/exp02_snr10_kin
