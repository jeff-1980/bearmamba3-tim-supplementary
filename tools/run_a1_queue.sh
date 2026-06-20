#!/usr/bin/env bash
# tools/run_a1_queue.sh
# A1 EAAI SNR grid: {-8,-6,-4,-2,0,+10} dB × {nokin,kin} × 5 seeds
# Run sequentially; writes mean±std to stdout after each pair.
set -e
cd "$(dirname "$0")/.."
source venv/bin/activate

TRAIN="experiments/exp01_cwru_baseline/train.py"
EXP_DIR="experiments/exp02_snr_ablation"

# Wait for any in-progress -5dB runs to finish (seed_4.json)
echo "=== Waiting for in-progress -5dB runs to finish ==="
until [[ -f results/exp02_snr-5_nokin/seed_4.json ]]; do sleep 30; done
echo "  -5dB nokin complete."
# Give kin a bit more time if needed
sleep 10

summarize() {
    python tools/summarize_ablation.py "results/$1/" 2>/dev/null || echo "  (no results yet)"
}

# Full EAAI grid
for snr in -8 -6 -4 -2 0 10; do
    tag="snr${snr}"
    echo ""
    echo "============================================================"
    echo "  A1: SNR=${snr}dB  CE-only (nokin)"
    echo "============================================================"
    python "$TRAIN" --config "${EXP_DIR}/config_${tag}_nokin.yaml"
    summarize "exp02_${tag}_nokin"

    echo ""
    echo "============================================================"
    echo "  A1: SNR=${snr}dB  +L_kin (λ=0.01)"
    echo "============================================================"
    python "$TRAIN" --config "${EXP_DIR}/config_${tag}_kin.yaml"
    summarize "exp02_${tag}_kin"

    echo "  *** SNR=${snr}dB pair complete ***"
done

echo ""
echo "============================================================"
echo "  A1 EAAI grid DONE — full summary:"
echo "============================================================"
python tools/summarize_ablation.py results/exp02_snr-8_nokin results/exp02_snr-8_kin \
    results/exp02_snr-6_nokin results/exp02_snr-6_kin \
    results/exp02_snr-4_nokin results/exp02_snr-4_kin \
    results/exp02_snr-2_nokin results/exp02_snr-2_kin \
    results/exp02_snr0_nokin results/exp02_snr0_kin \
    results/exp02_snr10_nokin results/exp02_snr10_kin 2>/dev/null

echo ""
echo "A1 queue complete. Ready for A2/A4."
