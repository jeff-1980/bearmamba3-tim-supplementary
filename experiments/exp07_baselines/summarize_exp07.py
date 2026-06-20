"""
汇总 exp07 所有基线结果，输出 LaTeX 表格行 + JSON 汇总。

用法:
  python experiments/exp07_baselines/summarize_exp07.py
"""
import json, sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS = PROJECT_ROOT / "results"

SNR_TAGS    = ["clean", "snr0", "snr-2", "snr-4", "snr-6", "snr-8"]
SNR_DISPLAY = ["Clean", "0", "−2", "−4", "−6", "−8"]
BACKBONES   = {
    "sksvm":         "SK-SVM",
    "cnn1d":         "1D-CNN",
    "transformer1d": "Transformer-1D",
    "mamba2":        "BearMamba-2",
    "mamba3_nokin":  "BearMamba-3 (ours)",
    "mamba3_kin":    "BearMamba-3 + $\\mathcal{L}_{\\text{kin}}$ (ours)",
}

# ── 从已有 A1/A3 结果读取 BM3 和 BM2 ────────────────────────────────────
BM3_NOKIN = {   # from CLAUDE.md A1 results
    "clean": (99.98, 0.04),
    "snr0":  (99.75, 0.22),
    "snr-2": (99.32, 0.35),
    "snr-4": (97.37, 1.01),
    "snr-6": (92.49, 1.95),
    "snr-8": (85.45, 2.90),
}
BM3_KIN = {
    "clean": (99.98, 0.04),
    "snr0":  (99.81, 0.13),
    "snr-2": (99.34, 0.24),
    "snr-4": (97.57, 0.38),
    "snr-6": (92.40, 1.68),
    "snr-8": (85.76, 2.03),
}
BM2_NOKIN = {   # from CLAUDE.md A3 results
    "clean": (100.00, 0.00),
    "snr0":  (99.86, 0.11),
    "snr-2": (99.51, 0.54),
    "snr-4": (98.81, 0.54),
    "snr-6": (95.79, 1.37),
    "snr-8": (91.03, 0.93),
}

def load_deep(name_prefix, snr_tag):
    """Load metrics.json or individual seed jsons from results dir."""
    d = RESULTS / f"{name_prefix}_{snr_tag}"
    metrics = d / "metrics.json"
    if metrics.exists():
        with open(metrics) as f:
            m = json.load(f)
        return m.get("mean_acc", 0)*100, m.get("std_acc", 0)*100
    # fallback: collect per-seed json
    seeds = sorted(d.glob("seed_*.json"))
    if not seeds:
        return None, None
    accs = []
    for s in seeds:
        with open(s) as f:
            j = json.load(f)
        accs.append(j.get("best_val_acc", j.get("acc", 0)) * 100)
    if len(accs) < 5:
        return None, None
    return float(np.mean(accs)), float(np.std(accs, ddof=1))

def load_sksvm(snr_tag):
    d = RESULTS / f"exp07_sksvm_{snr_tag}"
    metrics = d / "metrics.json"
    if metrics.exists():
        with open(metrics) as f:
            m = json.load(f)
        return m["mean_acc"]*100, m["std_acc"]*100
    seeds = sorted(d.glob("seed*.json"))
    if len(seeds) < 5:
        return None, None
    accs = [json.load(open(s))["acc"]*100 for s in seeds]
    return float(np.mean(accs)), float(np.std(accs, ddof=1))

def fmt(mean, std):
    if mean is None:
        return "—"
    return f"${mean:.2f}\\pm{std:.2f}$"

def main():
    rows = {}

    # SK-SVM
    rows["sksvm"] = {}
    for snr in SNR_TAGS:
        rows["sksvm"][snr] = load_sksvm(snr)

    # CNN1D + Transformer1D
    for bb in ("cnn1d", "transformer1d"):
        rows[bb] = {}
        for snr in SNR_TAGS:
            rows[bb][snr] = load_deep(f"exp07_{bb}", snr)

    # BM2, BM3 from hardcoded CLAUDE.md values
    rows["mamba2"]       = {k: BM2_NOKIN[k] for k in SNR_TAGS}
    rows["mamba3_nokin"] = {k: BM3_NOKIN[k] for k in SNR_TAGS}
    rows["mamba3_kin"]   = {k: BM3_KIN[k]   for k in SNR_TAGS}

    # ── Print LaTeX table ────────────────────────────────────────────
    header_cols = " & ".join(["Method"] + [f"${d}$~dB" for d in SNR_DISPLAY])
    print("% ── Table: CWRU 4-class baseline comparison ───────────────────────")
    print(r"\begin{table}[!t]")
    print(r"\centering")
    print(r"\caption{CWRU 4-class accuracy (\%) for all methods across EAAI SNR grid (mean$\pm$std, $n=5$ seeds). Clean: no noise.}")
    print(r"\label{tab:cwru_all_baselines}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(f"Method & Clean & 0~dB & $-2$~dB & $-4$~dB & $-6$~dB & $-8$~dB \\\\")
    print(r"\midrule")

    for key, label in BACKBONES.items():
        if key not in rows:
            continue
        cells = []
        for snr in SNR_TAGS:
            v = rows[key].get(snr, (None, None))
            cells.append(fmt(*v))
        sep = r"\midrule" if key == "mamba2" else ""
        if sep:
            print(sep)
        print(f"{label} & " + " & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    # ── Print plain summary ─────────────────────────────────────────
    print("\n\n── Plain summary ──")
    for key, label in BACKBONES.items():
        if key not in rows:
            continue
        line = f"{label:<35}"
        for snr in SNR_TAGS:
            v = rows[key].get(snr, (None, None))
            if v[0] is None:
                line += "   — "
            else:
                line += f"  {v[0]:6.2f}"
        print(line)

if __name__ == "__main__":
    main()
