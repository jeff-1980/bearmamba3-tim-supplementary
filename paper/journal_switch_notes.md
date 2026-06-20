# Journal Switch Notes: elsarticle → IEEEtran (D34, 2026-06-18)

> 自动化执行记录：从 MSSP (Elsevier elsarticle) 切换到 IEEE TIM (IEEEtran)
> 决策依据：CLAUDE.md 顶部 D34 段 + 规划/总体规划与决策日志.md D34 行

---

## 模板层级改动一览

| 改动项 | elsarticle (前) | IEEEtran (后) |
|---|---|---|
| documentclass | `[review,12pt]{elsarticle}` | `[journal,11pt,a4paper,draftcls,onecolumn]{IEEEtran}` |
| Journal 声明 | `\journal{Mechanical Systems and Signal Processing}` | （删除，IEEEtran 不需要）|
| Bibliography style | `elsarticle-num-names` | `IEEEtran` |
| 作者块 | `\author[inst1]{...}` + `\affiliation[inst1]{...}` + `\cortext` + `\ead{...}` | `\author{...}` + `\thanks{...}` 含通讯信息和 email |
| Frontmatter 容器 | `\begin{frontmatter}...\end{frontmatter}` | （删除，改用 `\maketitle`） |
| Abstract 关键词 | `\begin{keyword}...\sep...\end{keyword}` | `\begin{IEEEkeywords}...,...\end{IEEEkeywords}` |
| Member 标注 | （elsarticle 无概念）| `\IEEEmembership{Member,~IEEE,}` |

---

## 编译验证

| 项目 | 结果 |
|---|---|
| `pdflatex main` × 3 + `bibtex main` | ✅ 全部 0 errors |
| 最终 PDF 页数 | 34 页 |
| 最终 PDF 大小 | 711,719 字节（vs 旧 elsarticle 版 807,485 字节，IEEE 单栏更紧凑）|
| 备份文件 | `main_tim_v1.pdf` |

---

## 警告归类（15 条）

| 类别 | 来源 | 是否需修复 |
|---|---|---|
| **未定义引用 `li2017revisiting`** | 预存 TODO（与 D34 无关）| ⚠️ 写作期补 BibTeX |
| **未定义引用 `peng2018domain`** | 预存 TODO（与 D34 无关）| ⚠️ 写作期补 BibTeX |
| **未定义引用 `tab:b2_per_snr`** | section4 标签未匹配 | ⚠️ 写作期校对标签 |
| Font shape `OT1/ptm/m/scit` | IEEEtran + Times 字体常见，仅是 small caps italic 退化 | 🟢 可忽略 |
| `\maketitle` 后内容浮动 | IEEEtran draftcls 自然 overfull/underfull | 🟢 接受 |
| `rerunfilecheck` 提示 | 多次 pass 后 .out 收敛，已多跑一次 | 🟢 解决 |

---

## 兼容性陷阱（实际遭遇）

| 陷阱 | 处置 |
|---|---|
| elsarticle 的 `\linenumbers` 在 frontmatter 内自动；IEEEtran 需在 `\maketitle` 后显式调用 | ✅ 已调整顺序 |
| `\begin{keyword}` 是 elsarticle 专属 | ✅ 切换为 `\begin{IEEEkeywords}`，`\sep` 改逗号 |
| `\affiliation{organization=..., city=...}` 不存在于 IEEEtran | ✅ 改用 `\thanks{Y. Wang is with ...}` 风格 |
| `\cortext[cor1]{Corresponding author}` 不存在 | ✅ 改用 `\thanks{... (corresponding author, e-mail: ...)}` |
| `\ead{...}` (Elsevier email 宏) 不存在 | ✅ 改用 `\thanks{... e-mail: ...}` |
| `threeparttable`, `booktabs`, `microtype` | 🟢 与 IEEEtran 完全兼容，未改 |
| `soul` (`\hl{}` 命令) | 🟢 与 IEEEtran 兼容 |
| `lineno` (`\linenumbers`) | 🟢 与 IEEEtran 兼容 |

---

## 仍待人工确认

- [ ] **作者 IEEE Membership 状态**：当前用 `\IEEEmembership{Member,~IEEE,}` 占位。若 Yi Wang 非 IEEE Member，改为空 `\IEEEmembership{}` 或删除整个域。
- [ ] **历史合并版本 `bearmamba3_merged.tex` 是否清理**：该 105KB 单文件版本仍在 paper/，仍用 elsarticle/MSSP。**不参与当前编译**（main.tex 不 `\input{bearmamba3_merged}`），但 grep MSSP/elsarticle 时会被误击中。建议改名为 `bearmamba3_merged_LEGACY_20260615.tex` 或移到 `paper/archive/`。

---

## 重新编译命令（参考）

```bash
cd ~/论文8/paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

---

*生成时间：2026-06-18，由 Claudian via UNC 远程编辑 + WSL pdflatex 执行*
