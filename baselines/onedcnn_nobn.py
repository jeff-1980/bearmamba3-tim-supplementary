"""
baselines/onedcnn_nobn.py — 1D-CNN 无 BatchNorm 消融版

与 cnn1d.py 完全相同，仅移除所有 BatchNorm1d 层。
用途：验证 BN 是否是 1D-CNN 在 CWRU 低 SNR 优势的根因（M_ext Phase 2b，BLOCK-P2-1 Option A）。

结构：
  - conv_embed：Conv1d, kernel=8, stride=conv_stride, padding=3（无 BN/GELU，与 cnn1d.py 一致）
  - 4 个双卷积块：Conv1d → GELU → Conv1d → GELU → MaxPool（去掉 BN）
  - 全局平均池化 → 线性分类头
"""
import torch.nn as nn


class BearCNN1D_NoBN(nn.Module):
    def __init__(self, d_model: int = 64, n_layers: int = 4,
                 n_sensors: int = 1, n_classes: int = 4,
                 conv_stride: int = 2, **kwargs):
        super().__init__()
        self.conv_embed = nn.Conv1d(n_sensors, d_model, kernel_size=8,
                                    stride=conv_stride, padding=3)
        blocks = []
        for _ in range(n_layers):
            blocks.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GELU(),
                nn.MaxPool1d(2),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x, return_kin: bool = False):
        x = self.conv_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(-1)
        logits = self.classifier(x)
        if return_kin:
            return logits, None
        return logits
