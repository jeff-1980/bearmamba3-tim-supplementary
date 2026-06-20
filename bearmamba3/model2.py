"""
bearmamba3/model2.py — BearMamba-2 骨干（A3 消融对照基线）

与 BearMamba-3 公平协议:
  - 完全相同的 conv_embed（非 Patch, 卷积嵌入）
  - 完全相同的 pre-norm residual 结构
  - Mamba-2 替换 Mamba-3（无 RoPE, 不支持 L_kin）
"""
import torch
import torch.nn as nn
from mamba_ssm.modules.mamba2 import Mamba2


class BearMamba2(nn.Module):
    def __init__(self, d_model=64, d_state=128, n_layers=4,
                 n_sensors=3, n_classes=4, conv_stride=2,
                 headdim=64, dtype=torch.bfloat16):
        super().__init__()
        self.conv_stride = conv_stride
        self.conv_embed = nn.Conv1d(n_sensors, d_model, kernel_size=7,
                                    stride=conv_stride, padding=3)
        self.mamba_layers = nn.ModuleList([
            Mamba2(d_model=d_model, d_state=d_state, headdim=headdim,
                   dtype=dtype)
            for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model, dtype=dtype) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)
        self.dtype = dtype

    def forward(self, x):
        """x: (B, n_sensors, L) → logits (B, n_classes)"""
        h = self.conv_embed(x.to(self.conv_embed.weight.dtype))
        h = h.transpose(1, 2).to(self.dtype)                     # (B, L', D)
        for layer, ln in zip(self.mamba_layers, self.layer_norms):
            h = h + layer(ln(h))
        h = self.norm(h.float()).mean(dim=1)
        return self.classifier(h)
