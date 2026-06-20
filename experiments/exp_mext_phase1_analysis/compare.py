"""
M_ext Phase 1 分析：1D-CNN vs BM3+kin 跨数据集对比

输出：
  results/exp_mext_phase1_analysis/summary.json
  results/exp_mext_phase1_analysis/comparison_table.csv
  终端打印决策门判据（情景 A/B/C）
"""
import json, csv, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path("~/论文8").expanduser()
RES  = ROOT / "results"


def load_per_seed(path, key="best_val_accs"):
    """从 summary.json 读 per-seed 列表（×100 → %）"""
    d = json.load(open(path))
    if key in d:
        return [x * 100 for x in d[key]]
    raise KeyError(f"{key} not in {path}: {list(d.keys())}")


def load_lobo_per_seed(path):
    """XJTU LOBO summary: per_seed_macro_recall"""
    d = json.load(open(path))
    return [x * 100 for x in d["per_seed_macro_recall"]]


def load_cross_per_seed(path):
    """XJTU cross summary: macro_f1s"""
    d = json.load(open(path))
    return [x * 100 for x in d["macro_f1s"]]


def load_e14_dual(snr_tag):
    """E1.4 CWRU dual: 从 seed_*.json 聚合"""
    d = RES / f"exp_mext_e14_1dcnn_cwru_dual_{snr_tag}"
    files = sorted(d.glob("seed_*.json"))
    if len(files) < 5:
        return None
    return [json.load(open(f))["best_val_acc"] * 100 for f in files]


def wilcoxon_p(a, b):
    """双侧 Wilcoxon signed-rank，n=5"""
    try:
        if np.allclose(a, b):
            return 1.0
        stat, p = wilcoxon(a, b, alternative="two-sided")
        return float(p)
    except Exception:
        return float("nan")


def fmt(vals):
    if vals is None:
        return "N/A", "N/A"
    return f"{np.mean(vals):.2f}", f"{np.std(vals, ddof=1):.2f}"


def main():
    rows = []

    # ── 1. PU 跨工况 ──────────────────────────────────────────────
    try:
        cnn_pu   = load_per_seed(RES / "exp_mext_e11_1dcnn_pu_cross/summary.json")
        bm3n_pu  = load_per_seed(RES / "exp06_pu_nokin/summary.json")
        bm3k_pu  = load_per_seed(RES / "exp06_pu_kin/summary.json")
        bm2_pu   = load_per_seed(RES / "exp06_pu_bm2_nokin/summary.json")
        p_vs_kin = wilcoxon_p(cnn_pu, bm3k_pu)
        delta    = np.mean(cnn_pu) - np.mean(bm3k_pu)
        rows.append({"dataset": "PU_cross",
                     "metric": "accuracy(%)",
                     "1D_CNN_mean": np.mean(cnn_pu),  "1D_CNN_std": np.std(cnn_pu, ddof=1),
                     "BM3_kin_mean": np.mean(bm3k_pu), "BM3_kin_std": np.std(bm3k_pu, ddof=1),
                     "BM3_nokin_mean": np.mean(bm3n_pu),
                     "BM2_mean": np.mean(bm2_pu),
                     "delta_cnn_vs_bm3kin": delta, "wilcoxon_p": p_vs_kin,
                     "cnn_per_seed": cnn_pu, "bm3kin_per_seed": bm3k_pu})
        print(f"\n[PU cross]  1D-CNN={np.mean(cnn_pu):.2f}±{np.std(cnn_pu,ddof=1):.2f}%"
              f"  BM3+kin={np.mean(bm3k_pu):.2f}±{np.std(bm3k_pu,ddof=1):.2f}%"
              f"  Δ={delta:+.2f}pp  p={p_vs_kin:.3f}")
    except FileNotFoundError as e:
        print(f"[PU cross]  MISSING: {e}")

    # ── 2. XJTU LOBO ──────────────────────────────────────────────
    try:
        cnn_lobo  = load_lobo_per_seed(RES / "exp_mext_e12_1dcnn_xjtu_lobo/summary.json")
        bm3n_lobo = load_lobo_per_seed(RES / "exp_xjtu_lobo_nokin/summary.json")
        bm3k_lobo = load_lobo_per_seed(RES / "exp_xjtu_lobo_kin/summary.json")
        p_vs_kin  = wilcoxon_p(cnn_lobo, bm3k_lobo)
        delta     = np.mean(cnn_lobo) - np.mean(bm3k_lobo)
        rows.append({"dataset": "XJTU_LOBO",
                     "metric": "macro_recall(%)",
                     "1D_CNN_mean": np.mean(cnn_lobo),  "1D_CNN_std": np.std(cnn_lobo, ddof=1),
                     "BM3_kin_mean": np.mean(bm3k_lobo), "BM3_kin_std": np.std(bm3k_lobo, ddof=1),
                     "BM3_nokin_mean": np.mean(bm3n_lobo),
                     "delta_cnn_vs_bm3kin": delta, "wilcoxon_p": p_vs_kin,
                     "cnn_per_seed": cnn_lobo, "bm3kin_per_seed": bm3k_lobo})
        print(f"[XJTU LOBO] 1D-CNN={np.mean(cnn_lobo):.2f}±{np.std(cnn_lobo,ddof=1):.2f}%"
              f"  BM3+kin={np.mean(bm3k_lobo):.2f}±{np.std(bm3k_lobo,ddof=1):.2f}%"
              f"  Δ={delta:+.2f}pp  p={p_vs_kin:.3f}")
    except FileNotFoundError as e:
        print(f"[XJTU LOBO] MISSING: {e}")

    # ── 3. XJTU 跨工况 ────────────────────────────────────────────
    try:
        cnn_cross  = load_cross_per_seed(RES / "exp_mext_e13_1dcnn_xjtu_cross/summary.json")
        bm3n_cross = load_cross_per_seed(RES / "exp_xjtu_cross_nokin/summary.json")
        bm3k_cross = load_cross_per_seed(RES / "exp_xjtu_cross_kin/summary.json")
        p_vs_kin   = wilcoxon_p(cnn_cross, bm3k_cross)
        delta      = np.mean(cnn_cross) - np.mean(bm3k_cross)
        rows.append({"dataset": "XJTU_cross",
                     "metric": "macro_F1(%)",
                     "1D_CNN_mean": np.mean(cnn_cross),  "1D_CNN_std": np.std(cnn_cross, ddof=1),
                     "BM3_kin_mean": np.mean(bm3k_cross), "BM3_kin_std": np.std(bm3k_cross, ddof=1),
                     "BM3_nokin_mean": np.mean(bm3n_cross),
                     "delta_cnn_vs_bm3kin": delta, "wilcoxon_p": p_vs_kin,
                     "cnn_per_seed": cnn_cross, "bm3kin_per_seed": bm3k_cross})
        print(f"[XJTU cross] 1D-CNN={np.mean(cnn_cross):.2f}±{np.std(cnn_cross,ddof=1):.2f}%"
              f"  BM3+kin={np.mean(bm3k_cross):.2f}±{np.std(bm3k_cross,ddof=1):.2f}%"
              f"  Δ={delta:+.2f}pp  p={p_vs_kin:.3f}")
    except FileNotFoundError as e:
        print(f"[XJTU cross] MISSING: {e}")

    # ── 4. CWRU dual-sensor SNR 扫描 ───────────────────────────────
    # B2 对照（单传感器 1D-CNN = exp07_cnn1d_snr*；双传感器 BM3 = exp_b2_dual_nokin_snr*）
    from glob import glob as _glob
    print("\n[CWRU dual-sensor SNR scan]")
    e14_rows = []
    for snr in [-8, -6, -4, -2, 0]:
        # E1.4 1D-CNN dual results dir uses literal minus: snr-8, snr-4, etc.
        e14_tag = f"snr{snr}"
        e14     = load_e14_dual(e14_tag)
        # 单传感器 1D-CNN baseline (from exp07)
        single_files = sorted((RES / f"exp07_cnn1d_snr{snr}").glob("seed_*.json"))
        single_cnn   = [json.load(open(f))["best_val_acc"] * 100 for f in single_files] if single_files else None
        # BM3 dual (B2): 0dB uses base dir, negative uses snrm{n} suffix
        if snr == 0:
            bm3_dual_d = RES / "exp_b2_dual_nokin"
        else:
            bm3_dual_d = RES / f"exp_b2_dual_nokin_snrm{abs(snr)}"
        bm3_dual = [json.load(open(f))["best_val_acc"] * 100
                    for f in sorted(bm3_dual_d.glob("seed_*.json"))] if bm3_dual_d.exists() else None

        if e14:
            delta_vs_bm3dual = np.mean(e14) - np.mean(bm3_dual) if bm3_dual else float("nan")
            delta_vs_single  = np.mean(e14) - np.mean(single_cnn) if single_cnn else float("nan")
            bm3str = f"{np.mean(bm3_dual):.2f}%" if bm3_dual else "N/A"
            print(f"  SNR={snr:+d}dB  CNN_dual={np.mean(e14):.2f}±{np.std(e14,ddof=1):.2f}%"
                  f"  BM3_dual={bm3str}"
                  f"  Δ_vs_BM3dual={delta_vs_bm3dual:+.2f}pp"
                  + (f"  Δ_vs_single={delta_vs_single:+.2f}pp" if single_cnn else ""))
            e14_rows.append({"snr": snr, "cnn_dual_mean": np.mean(e14),
                             "cnn_dual_std": np.std(e14, ddof=1),
                             "bm3_dual_mean": np.mean(bm3_dual) if bm3_dual else None,
                             "cnn_single_mean": np.mean(single_cnn) if single_cnn else None,
                             "wilcoxon_p_vs_bm3dual": wilcoxon_p(e14, bm3_dual) if bm3_dual else None})
        else:
            print(f"  SNR={snr:+d}dB  MISSING")

    rows_for_csv = rows  # XJTU/PU rows

    # ── 决策门判据 ─────────────────────────────────────────────────
    if rows:
        deltas = [r["delta_cnn_vs_bm3kin"] for r in rows]
        mean_delta = np.mean(deltas)
        all_positive = all(d > 0 for d in deltas)
        all_ge3 = all(d >= 3.0 for d in deltas)
        all_le2 = all(d <= 2.0 for d in deltas)
        any_negative = any(d < 0 for d in deltas)
        any_le_minus3 = any(d <= -3.0 for d in deltas)

        print(f"\n{'='*60}")
        print(f"决策门判据  Δ per dataset: {[f'{d:+.2f}pp' for d in deltas]}")
        print(f"           均值 Δ = {mean_delta:+.2f}pp")
        if all_ge3:
            print("→ 情景 A：1D-CNN 全部 ≥+3pp — ⚠️  STOP，咨询用户")
        elif any_le_minus3:
            print("→ 情景 C：存在 ≤−3pp 崩溃 — ✅ 进 Phase 2，双优叙事")
        elif all_le2 or any_negative:
            print("→ 情景 B：1D-CNN ≤+2pp 或负 — ✅ 进 Phase 2，完成闭环")
        else:
            print("→ 混合情景：部分 Δ>2pp 但未全 ≥+3pp，请用户决策")
        print(f"{'='*60}")

    # ── 落盘 ──────────────────────────────────────────────────────
    out_dir = RES / "exp_mext_phase1_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"phase1_rows": rows, "e14_cwru_dual": e14_rows}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    csv_path = out_dir / "comparison_table.csv"
    if rows:
        fields = ["dataset", "metric", "1D_CNN_mean", "1D_CNN_std",
                  "BM3_kin_mean", "BM3_kin_std", "delta_cnn_vs_bm3kin", "wilcoxon_p"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    print(f"\nSaved → {out_dir}/summary.json")
    print(f"Saved → {csv_path}")


if __name__ == "__main__":
    main()
