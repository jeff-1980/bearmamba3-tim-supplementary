#!/bin/bash
# experiments/exp_xjtu/run_b4_5.sh
# B4-5: XJTU-SY dual-channel (horizontal+vertical) vs single-channel (D28)
#
# Protocol (D28):
#   主赛道：LOBO Cond3, {单,双}通道 × {±L_kin} × 5 seeds × 50 epochs
#   佐证赛道：Cond2→Cond3 跨工况, {单,双}通道 × {±L_kin} × 5 seeds
#   单通道结果：复用 B4-3 results (SKIP if summary.json exists)
#   双通道结果：4 新实验 (~8h total)
#
# C3 成功判据 (D28)：
#   双通道在 CWRU(B2) 和 XJTU(B4-5) 两处均呈一致正向增益；
#   CWRU -4dB nokin p=0.031 已满足；B4-5 LOBO 若 Wilcoxon p<0.05 则达最强等级。
#
# 用法:
#   tmux new -s b4_5
#   source venv/bin/activate
#   bash experiments/exp_xjtu/run_b4_5.sh 2>&1 | tee results/b4_5.log

set -e
cd /home/jeffwork/论文8
source venv/bin/activate

TRAIN="experiments/exp_xjtu/train.py"

run_if_needed() {
    local cfg="$1"
    local results_dir="$2"
    if [ -f "${results_dir}/summary.json" ]; then
        echo "[SKIP] ${results_dir} already complete"
        return
    fi
    echo ""
    echo "========================================================"
    echo "[RUN ] $cfg"
    echo "========================================================"
    python "$TRAIN" --config "$cfg"
}

echo "========================================================"
echo "  B4-5 Dual-Channel Training Queue (D28)"
echo "  $(date)"
echo "========================================================"

# ── LOBO single-channel (SKIP if B4-3 already done) ───────────────
echo ""
echo "── LOBO single-channel (reuse B4-3) ──"
run_if_needed \
    experiments/exp_xjtu/config_lobo_nokin.yaml \
    ~/论文8/results/exp_xjtu_lobo_nokin

run_if_needed \
    experiments/exp_xjtu/config_lobo_kin.yaml \
    ~/论文8/results/exp_xjtu_lobo_kin

# ── LOBO dual-channel (NEW) ────────────────────────────────────────
echo ""
echo "── LOBO dual-channel (NEW) ──"
run_if_needed \
    experiments/exp_xjtu/config_lobo_dual_nokin.yaml \
    ~/论文8/results/exp_xjtu_lobo_dual_nokin

run_if_needed \
    experiments/exp_xjtu/config_lobo_dual_kin.yaml \
    ~/论文8/results/exp_xjtu_lobo_dual_kin

# ── Cross-condition single-channel (SKIP if B4-3 done) ────────────
echo ""
echo "── Cross single-channel (reuse B4-3) ──"
run_if_needed \
    experiments/exp_xjtu/config_cross_nokin.yaml \
    ~/论文8/results/exp_xjtu_cross_nokin

run_if_needed \
    experiments/exp_xjtu/config_cross_kin.yaml \
    ~/论文8/results/exp_xjtu_cross_kin

# ── Cross-condition dual-channel (NEW) ────────────────────────────
echo ""
echo "── Cross dual-channel (NEW) ──"
run_if_needed \
    experiments/exp_xjtu/config_cross_dual_nokin.yaml \
    ~/论文8/results/exp_xjtu_cross_dual_nokin

run_if_needed \
    experiments/exp_xjtu/config_cross_dual_kin.yaml \
    ~/论文8/results/exp_xjtu_cross_dual_kin

echo ""
echo "========================================================"
echo "  B4-5 ALL DONE  $(date)"
echo "========================================================"

# ── Quick summary ──────────────────────────────────────────────────
python3 experiments/exp_xjtu/analyze_b4_5.py
