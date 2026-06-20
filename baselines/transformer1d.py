"""
baselines/transformer1d.py — 1D Transformer 基线（~201K 参数）

与 BearMamba-3 的公平对照协议:
  - 相同 conv_embed（Conv1d, kernel=8, stride=conv_stride, padding=3，无 BN/GELU）
  - 4 × Pre-Norm TransformerEncoderLayer（d_model=64, nhead=4, dim_ff=256）
  - 全局平均池化 → 线性分类头
  - 无 Patch tokenization（与本组嵌入约定一致）
  - 不支持 return_kin（纯 CE 基线）
"""
import torch
import torch.nn as nn
import math


class BearTransformer1D(nn.Module):
    def __init__(self, d_model: int = 64, n_layers: int = 4,
                 n_sensors: int = 1, n_classes: int = 4,
                 conv_stride: int = 2, nhead: int = 4,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        self.conv_embed = nn.Conv1d(n_sensors, d_model, kernel_size=8,
                                    stride=conv_stride, padding=3)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # Pre-Norm（与 BM3 pre-norm residual 一致）
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, return_kin: bool = False):
        x = self.conv_embed(x)              # (B, d, L/stride)
        x = x.permute(0, 2, 1)             # (B, L, d)
        x = self.transformer(x)             # (B, L, d)
        x = self.norm(x).mean(1)            # (B, d) global avg pool
        logits = self.classifier(x)
        if return_kin:
            return logits, None
        return logits
