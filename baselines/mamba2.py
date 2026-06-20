"""
baselines/mamba2.py — BearMamba-2 骨干（A3 消融对照，Mamba-2 vs Mamba-3）

公平对照协议（与 BearMamba-3 完全一致）:
  - conv_embed: Conv1d(kernel=7, stride=conv_stride, padding=3)，无 BN/GELU
    EAAI 原版有 BN+GELU (common.py)；我们移除以隔离骨干差异。
  - Pre-norm residual: h = h + Mamba2(LayerNorm(h))
  - 输出: 全局平均池化 → Linear 分类头

不同于 BearMamba-3 的地方:
  - Mamba2 层（mamba_ssm.modules.mamba2.Mamba2），无 RoPE，无 theta/DT 激活
  - 不支持 return_kin=True（L_kin 消融专用对照，不施加运动学正则）

来源：从 /home/jeffwork/论文1/src/models/mamba2.py 移植骨干层；
      嵌入层、pre-norm residual 改为与 bearmamba3/model.py 一致。
      拷贝时间：2026-06-10 22:00:33
"""
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba2 import Mamba2


class BearMamba2(nn.Module):
    def __init__(self, d_model: int = 64, d_state: int = 128, n_layers: int = 4,
                 n_sensors: int = 1, n_classes: int = 4, conv_stride: int = 2,
                 headdim: int = 64, dtype=torch.bfloat16):
        super().__init__()
        self.conv_stride = conv_stride
        # 与 BearMamba-3 完全相同的 conv_embed（无 BN/GELU）
        self.conv_embed = nn.Conv1d(n_sensors, d_model, kernel_size=7,
                                    stride=conv_stride, padding=3)
        self.mamba_layers = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state, headdim=headdim, dtype=dtype)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model, dtype=dtype) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        self.dtype = dtype

    def forward(self, x, return_kin=False):
        """x: (B, n_sensors, L) → logits (B, n_classes)
        return_kin ignored (Mamba-2 has no RoPE theta/DT, L_kin unsupported)."""
        h = self.conv_embed(x.to(self.conv_embed.weight.dtype))
        h = h.transpose(1, 2).to(self.dtype)           # (B, L', D)
        for layer, ln in zip(self.mamba_layers, self.layer_norms):
            h = h + layer(ln(h))
        h = self.norm(h.float()).mean(dim=1)
        logits = self.classifier(h)
        return (logits, []) if return_kin else logits
