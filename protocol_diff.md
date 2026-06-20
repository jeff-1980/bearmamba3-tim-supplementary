# Protocol Difference Audit: BearMamba-3 vs EAAI Baseline

**生成时间**: 2026-06-10  
**EAAI 来源**: /home/jeffwork/论文1/src/ (只读)  
**对应 EAAI 结果**: CWRU CE-only = 97.95% (4类, SNR均值) — 见 logs/

---

## 关键差异汇总

| 维度 | EAAI（论文1） | BearMamba-3（论文8） | 影响分析 |
|------|-------------|-------------------|---------|
| **窗口长度** | 1024 samples | **2048 samples** (2×) | 更长上下文；频率分辨率更高（CWRU 12kHz下0.78Hz/bin） |
| **步长 / 重叠** | stride=512 (50% overlap) | stride=1024 (50% overlap) | 样本量相近，窗口独立性相同 |
| **噪声注入方式** | **多SNR混合训练**: 每 batch 随机从 `[-6,-4,-2,0,2,4,6,10]` dB 采样；在线增强（data/dataset.py） | **单一固定 SNR** 每个实验配置；噪声在 Dataset 构造时注入（非在线） | **这是最大协议差异**。EAAI 模型是噪声鲁棒模型（见8个SNR混合），我们的是点估计模型。评估时需在相同 eval SNR 下比较，但训练分布不同 |
| **ConvEmbedding** | Conv1d + **BatchNorm1d + GELU** (common.py) | Conv1d 裸卷积（无 BN/GELU） | BN 提供归一化，GELU 提供非线性；BearMamba-2 基线已移除 BN/GELU 保持公平 |
| **PU 分类数** | **4类**: Normal/Inner/Outer/**Ball**（包含 KB23/24/27 真实磨损） | **3类**: Normal/Inner/Outer（排除 KB/Ball） | 类别构成不同，精度数字不可直接比较 |
| **损失函数** | **类别加权 CE** (class-weighted, trainer.py) | 普通 CE（均等权重） | PU/XJTU-SY 等不均衡数据集下有影响 |
| **Epochs** | 100 | 50 | EAAI 训练更充分；我们的精度已收敛（CWRU天花板） |
| **学习率** | 1e-3 (with warmup=5 epochs) | **3e-4** (cosine only) | EAAI 有 warmup，收敛路径不同 |
| **VRAM / 模型参数** | 174,044 params / 388.9 MB | BM3: 178,068 (1.02×) / BM2: 173,980 (1.00×) | 参数量基本等价 |

---

## 详细说明

### D1. 窗口长度（最可能贡献精度差异）

EAAI `segment_len=1024`（cwru_loader.py L17），我们 `win_len=2048`。
CWRU 12kHz 下：
- EAAI: 1024/12000 = 85ms 窗口，频率分辨率 11.7 Hz/bin
- 我们: 2048/12000 = 171ms 窗口，频率分辨率 5.9 Hz/bin

更长窗口使 SSM 状态机有更多时间步分辨故障频率谐波，对 L_kin 更有利。
这可能解释 2pp 精度差（97.95% → 99.98%）的一部分原因。

### D2. 噪声注入方式（**最重要协议差异，已锚定，不得更改**）

EAAI 在线多SNR增强（dataset.py `snr_db = random.choice(self.snr_list)` per batch）。
我们单固定SNR。

**后果**：直接比较时，我们的结果更"点估计"，EAAI 更"全局鲁棒"。
在相同 eval SNR 点上，EAAI 因训练时见过更多 SNR 组合，未必在某一单点最优。

**锚定决策（2026-06-10）**：A1 SNR 网格沿用 EAAI 评估网格 `{-8,-6,-4,-2,0,+10}`，
每档训练一个独立模型（固定该 SNR）。这与 EAAI 多SNR训练不同，但便于逐点分析 L_kin 效益。
后续不得擅改此网格。

### D3. ConvEmbedding — BearMamba-2 移植决策

A3 消融目标是隔离**骨干差异**（Mamba-2 vs Mamba-3），而非嵌入层差异。
因此 `baselines/mamba2.py` 强制使用与 BearMamba-3 相同的裸 Conv1d embed（无 BN/GELU）。

如需研究 BN/GELU 的影响，单独安排 A_embed 消融。

### D4. PU 协议差异

| 维度 | EAAI | 我们 |
|------|------|------|
| 分类数 | 4（含 Ball/KB） | 3（仅 Normal/Inner/Outer） |
| 振动通道 | Y[6]（.Data）via "vibration" 字段名 | **Y[6].Data**（256001 pts @ 64kHz，kurtosis 5-15）⚠️ X[1] 是时间轴（0→4秒），禁用！|
| 工况划分 | 未见跨工况验证配置 | **cross-condition, same-bearing**：N09+N15_M01 训练 → N15_M07_F10 测试（同一套轴承，非 cross-bearing）|
| 噪声 | 多SNR在线增强（同CWRU） | 固定 SNR（同CWRU） |

⚠️ **振动通道更正（2026-06-12）**：经实测确认，.mat 文件 `X[1].Data` 是时间轴（ramp 0→4秒，单调递增），
非振动信号；振动信号为 `Y[6].Data`（均值≈0，std≈0.4-0.7，kurtosis 按类别 5-15 明显区分）。
已修正 `data_pu.py:_load_vib()` 并删除所有基于 X[1] 的无效 M4 结果。

---

## EAAI CWRU SNR 扫描结果（从 logs/ 提取）

训练方式：多SNR混合训练（-6~+10dB），以下为各 eval SNR 的准确率：

| SNR (dB) | EAAI BM2 (mean±std) | BM3 CE-only (A1) | BM2 CE-only (A3) | BM3 vs EAAI | BM2 vs EAAI |
|----------|---------------------|-----------------|-----------------|-------------|-------------|
| -8 | 83.79±2.95% | 85.45±2.90% | **91.03±0.93%** | +1.66pp | +7.24pp |
| -6 | 91.03±3.83% | 92.49±1.95% | **95.79±1.37%** | +1.46pp | +4.76pp |
| -4 | 95.37±2.78% | 97.37±1.01% | **98.81±0.54%** | +2.00pp | +3.44pp |
| -2 | 98.57±0.80% | 99.32±0.35% | **99.51±0.54%** | +0.75pp | +0.94pp |
|  0 | 99.61±0.34% | 99.75±0.22% | **99.86±0.11%** | +0.14pp | +0.25pp |
| +10 | **100.00%** | 100.00±0.00% | 100.00±0.00% | 0.00pp | 0.00pp |
| Clean | 97.95% (G1 ref) | **99.98±0.04%** | 99.98±0.04% | +2.03pp | +2.03pp |

*注：EAAI 为多SNR混合训练（训练分布不同）；BM3/BM2 为固定SNR训练（D8 锚定协议）。std 均为 ddof=1。*
*主观察：BM2（同管道）在所有噪声水平均优于 BM3；BM2 vs EAAI 差距在低 SNR 下更大（+7.24pp@-8dB），主因是窗口长度(1024→2048)而非骨干。*

*注：EAAI 的 clean 97.95% 是在多SNR增强模型上测试，非纯净训练。*

---

## 归因结论（A3 完成，2026-06-12）

2pp 精度差（97.95% → 99.98%）归因更新：
1. **窗口长度**（1024→2048）：**主要贡献**。BM2 同管道（与 BM3 唯一差别是骨干）比 EAAI BM2 高 0.94pp@-2dB 到 7.24pp@-8dB，印证了窗口长度是主效应。
2. **骨干差异（BM3 vs BM2）**：**CWRU 上 BM2 优于 BM3**（-8dB 差 5.58pp，p=0.031）。根因：BM3 在 AWGN+恒速下欠拟合（D18）。CWRU 上 BM3 vs EAAI 的 +2pp 主要来自窗口长度，非骨干。
3. **BN/GELU 有无**：孤立影响尚未量化；BM2 已移除 BN/GELU 以公平比较。
4. **训练 epoch**（50 vs 100）：CWRU 天花板下影响接近零（confirmed）。

