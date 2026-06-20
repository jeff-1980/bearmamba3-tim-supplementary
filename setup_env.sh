#!/usr/bin/env bash
# BearMamba-3 环境搭建（WSL Ubuntu + RTX A5000, sm_86）
# 用法: bash setup_env.sh 2>&1 | tee setup.log
set -e

echo "=== [1/5] 检查 CUDA 环境 ==="
nvidia-smi || { echo "未检测到 GPU，请确认 WSL CUDA 驱动"; exit 1; }
python3 --version

echo "=== [2/5] 创建虚拟环境 ==="
python3 -m venv ~/论文8/venv
source ~/论文8/venv/bin/activate
pip install --upgrade pip

echo "=== [3/5] 安装 PyTorch (CUDA 12.x) ==="
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "=== [4/5] 安装 Mamba (含 Mamba-3 + TileLang MIMO kernel) ==="
pip install packaging ninja einops "triton>=3.5.0"
pip install causal-conv1d
# tilelang 是 MIMO 的硬依赖；若此步失败，先只跑 SISO（见侦察报告结论五）
pip install tilelang==0.1.8 || echo "⚠️ tilelang 安装失败 → MIMO 暂不可用，SISO 不受影响"
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir \
    git+https://github.com/state-spaces/mamba.git --no-build-isolation

echo "=== [5/5] 项目依赖 ==="
pip install numpy scipy scikit-learn matplotlib pandas tqdm

echo ""
echo "✅ 完成。下一步: python verify_mamba3.py"
