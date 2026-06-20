# PG-TMT v3 跨速结果核对（REC-PGTMT-4）

> 关联：CLAUDE.md "## 写作必引同期竞品" 段 + REC-PGTMT-4 待办
> 核对来源：https://arxiv.org/html/2601.21293v3（v3 终版，2026-06-10）
> 核对时间：2026-06-18

---

## 核对问题

> "PG-TMT v3 是否在 Paderborn (PU) 上独立报告 1500rpm→900rpm 跨速结果（具体数字）？"

## 核对结论

❌ **PG-TMT v3 仅给出定性跨速论述，无 PU 1500→900rpm 精度数字表。**

### 证据（v3 HTML 摘录）

```
"Load and speed shifts. Order-band-aligned spectral attention stabilizes
defect-order harmonics under rpm changes, leading to high AUC retention
and limited drift in MTTD."
```

- 仅提供 **定性论述** ("high AUC retention", "limited drift")
- **无具体数字表** 列出 PU 1500→900rpm 或类似跨速场景的精度/F1
- Figure 6 / Figure 8 标示存在但**数值不可机读**（图像数据，未在 HTML 文本中转录）

### 其他相关发现

PG-TMT v3 的实验 section **整体偏向以下倾向**：

| 项 | 内容 |
|---|---|
| 比较基线 | VibrMamba、group Mamba、BMTM-net、Shrinkage Mamba、DANN、迁移学习——**未列 1D-CNN、WDCNN、ResNet 等经典基线** |
| 精度数字呈现 | 主要为 PR-AUC 趋势曲线，**少有具体表格数字** |
| 卖点定位 | reliability calibration（EVT）+ edge 部署（< 1MB）+ 可解释性（频域注意力对齐）——**明确不主打"打榜精度"** |
| BatchNorm | 架构描述无 BN 提及（与 BearMamba-3 一致） |

---

## 对 BearMamba-3 论文写作的影响

### 不需要补充对位说明
由于 PG-TMT 不提供 PU 跨速具体数字，BearMamba-3 的跨工况叙事（D29 + E1.3：BM3+kin 79.73% vs 1D-CNN 41.31% on XJTU cross）**无需配对说明**，原始论述完全立得住。

### Discussion 段已恰当处理
section5_discussion.tex 中两处 PG-TMT 引用均为**定性引用**（非数字对比），与 PG-TMT 原文呈现风格匹配：
- L128 "PG-TMT similarly sidesteps direct comparison against classical 1D-CNN baselines"
- L181 "PG-TMT achieves edge-IoT deployment at sub-1\,MB"

### section4 对比表保持不变
按 REC-PGTMT-2 决策：**禁止伪造 PG-TMT 数字进对比表**。本核对结论确认了这一决策的正确性——即使想伪造，原始论文也无精度表可参照。

---

## REC-PGTMT-4 状态

✅ **关闭**——本核对完成时已确认 PG-TMT v3 无可对位的跨速数字；BearMamba-3 论文的跨工况叙事不受 PG-TMT 影响，无需补充说明。

---

*核对人：Claudian via WebFetch on arxiv.org/html/2601.21293v3*
*下次核对触发条件：PG-TMT 升级到 v4 或被 IEEE IoT Journal 录用并发布详细数字时*
