#!/usr/bin/env bash
# tools/run_a3_a2_a4_queue.sh
# Continuation queue: A3 (BearMamba-2 × EAAI grid) → A2 (10-class) → A4 (λ scan)
# Waits for A1 queue to finish (all 12 result summaries present), then runs serially.
set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

TRAIN="experiments/exp01_cwru_baseline/train.py"

summarize() { python tools/summarize_ablation.py "results/$1/" 2>/dev/null || true; }

# ─── Wait for A1 to finish ────────────────────────────────────────────────────
echo "=== Waiting for A1 queue to finish (checking for snr10_kin summary) ==="
until [[ -f results/exp02_snr10_kin/summary.json ]]; do sleep 60; done
echo "  A1 queue complete."

# ─── A3: BearMamba-2 CE-only × EAAI SNR grid ─────────────────────────────────
echo ""
echo "============================================================"
echo "  A3: BearMamba-2 CE-only EAAI SNR grid"
echo "============================================================"
for snr in -8 -6 -4 -2 0 10; do
    echo ""
    echo "  A3: SNR=${snr}dB  BearMamba-2"
    python "$TRAIN" --config "experiments/exp04_mamba2/config_snr${snr}.yaml"
    summarize "exp04_mamba2_snr${snr}"
done
echo ""
echo "  *** A3 complete — BearMamba-2 SNR summary:"
python tools/summarize_ablation.py \
    results/exp04_mamba2_snr-8 results/exp04_mamba2_snr-6 \
    results/exp04_mamba2_snr-4 results/exp04_mamba2_snr-2 \
    results/exp04_mamba2_snr0  results/exp04_mamba2_snr10 2>/dev/null

# ─── A2: CWRU 10-class SNR=0 ±L_kin ─────────────────────────────────────────
echo ""
echo "============================================================"
echo "  A2: CWRU 10-class SNR=0dB  CE-only → +L_kin"
echo "============================================================"
python "$TRAIN" --config experiments/exp03_10class/config_snr0_nokin.yaml
summarize exp03_10class_snr0_nokin
python "$TRAIN" --config experiments/exp03_10class/config_snr0_kin.yaml
summarize exp03_10class_snr0_kin
echo "  *** A2 complete"

# ─── A4: λ scan @ SNR=0 ──────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  A4: λ scan {1e-3, 1e-2, 1e-1, 1.0} @ SNR=0dB"
echo "============================================================"
for cfg in config_lk1em3 config_lk1em2 config_lk1em1 config_lk1p0; do
    echo "  A4: $cfg"
    python "$TRAIN" --config "experiments/exp05_lambda_scan/${cfg}.yaml"
    name=$(python3 -c "import yaml; d=yaml.safe_load(open('experiments/exp05_lambda_scan/${cfg}.yaml')); print(d['name'])")
    summarize "$name"
done
echo ""
echo "  *** A4 complete — λ scan summary:"
python tools/summarize_ablation.py \
    results/exp05_lk1em3 results/exp05_lk1em2 \
    results/exp05_lk1em1 results/exp05_lk1p0 2>/dev/null

# ─── Final M3 summary ─────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  M3 FULL SUMMARY"
echo "============================================================"
echo "--- A1 BearMamba-3 EAAI grid ---"
python tools/summarize_ablation.py \
    results/exp02_snr-8_nokin results/exp02_snr-8_kin \
    results/exp02_snr-6_nokin results/exp02_snr-6_kin \
    results/exp02_snr-4_nokin results/exp02_snr-4_kin \
    results/exp02_snr-2_nokin results/exp02_snr-2_kin \
    results/exp02_snr0_nokin  results/exp02_snr0_kin \
    results/exp02_snr10_nokin results/exp02_snr10_kin 2>/dev/null
echo "--- A3 BearMamba-2 EAAI grid ---"
python tools/summarize_ablation.py \
    results/exp04_mamba2_snr-8 results/exp04_mamba2_snr-6 \
    results/exp04_mamba2_snr-4 results/exp04_mamba2_snr-2 \
    results/exp04_mamba2_snr0  results/exp04_mamba2_snr10 2>/dev/null
echo "--- A2 10-class ---"
python tools/summarize_ablation.py results/exp03_10class_snr0_nokin results/exp03_10class_snr0_kin 2>/dev/null
echo "--- A4 λ scan ---"
python tools/summarize_ablation.py \
    results/exp05_lk1em3 results/exp05_lk1em2 \
    results/exp05_lk1em1 results/exp05_lk1p0 2>/dev/null

echo ""
echo "M3 queue complete. Run /checkpoint M3 to gate-review before M4."
