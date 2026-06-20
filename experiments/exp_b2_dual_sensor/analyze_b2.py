#!/usr/bin/env python3
"""
experiments/exp_b2_dual_sensor/analyze_b2.py

B2 双传感器 vs 单传感器 — 跨 SNR 配对检验 + 曲线分析（D26）

主检验: 以 SNR 档为配对因子，25 个配对差值（5 SNR × 5 seeds）
  → Wilcoxon signed-rank (non-parametric) + one-sample t-test
  → 报告系统性偏移（mean Δ ± se）和 p 值
每点统计: 每 SNR 点单独 Wilcoxon（5 seeds），作为辅助表格
混合效应: 若 pingouin/statsmodels 可用，追加 LMM 结果
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent   # ~/论文8
SNR_GRID = [-8, -6, -4, -2, 0]

# ── 结果路径映射 ─────────────────────────────────────────────────────────────

def snr_tag(snr): return f"snrm{abs(int(snr))}" if snr < 0 else f"snr{int(snr)}"


def result_path(sensor: str, kind: str, snr: float) -> Path:
    """sensor='single'|'dual', kind='nokin'|'kin'"""
    if sensor == "single":
        # A1 dirs named exp02_snr{-8,-6,...,0,10}_{kind} (literal sign in name)
        name = f"exp02_snr{int(snr)}_{kind}"
    else:
        if snr == 0:
            name = f"exp_b2_dual_{kind}"
        else:
            name = f"exp_b2_dual_{kind}_{snr_tag(snr)}"
    return ROOT / "results" / name / "summary.json"


def load_accs(sensor: str, kind: str, snr: float) -> np.ndarray | None:
    p = result_path(sensor, kind, snr)
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return np.array(d["best_val_accs"])


# ── 主分析 ──────────────────────────────────────────────────────────────────

def main():
    from scipy.stats import wilcoxon, ttest_1samp, levene

    # Per-SNR stats table
    print("\n" + "=" * 80)
    print("  Per-SNR: Single vs Dual (CE-only) and Single+kin vs Dual+kin")
    print("=" * 80)
    header = f"{'SNR':>5}  {'S_nokin':>12}  {'D_nokin':>12}  Δ_nokin  "
    header += f"{'S_kin':>12}  {'D_kin':>12}  Δ_kin"
    print(header)
    print("-" * 80)

    all_deltas_nokin = []   # (n_snr × n_seeds,) for cross-SNR test
    all_deltas_kin   = []

    per_snr = {}
    for snr in SNR_GRID:
        s_nk = load_accs("single", "nokin", snr)
        d_nk = load_accs("dual",   "nokin", snr)
        s_k  = load_accs("single", "kin",   snr)
        d_k  = load_accs("dual",   "kin",   snr)

        if any(x is None for x in [s_nk, d_nk, s_k, d_k]):
            missing = [t for t, x in [("S_nokin",s_nk),("D_nokin",d_nk),("S_kin",s_k),("D_kin",d_k)] if x is None]
            print(f"  SNR={snr:3d}dB  [MISSING: {', '.join(missing)}]")
            continue

        dn_nk = d_nk - s_nk   # per-seed delta, nokin
        dn_k  = d_k  - s_k    # per-seed delta, kin
        all_deltas_nokin.append(dn_nk)
        all_deltas_kin.append(dn_k)
        per_snr[snr] = (s_nk, d_nk, s_k, d_k, dn_nk, dn_k)

        def fmt(a): return f"{a.mean()*100:6.2f}±{a.std(ddof=1)*100:.2f}%"
        def dlt(d): return f"{d.mean()*100:+.2f}pp"
        print(f"  {snr:3d}dB  {fmt(s_nk)}  {fmt(d_nk)}  {dlt(dn_nk)}   "
              f"{fmt(s_k)}  {fmt(d_k)}  {dlt(dn_k)}")

    print()

    if not all_deltas_nokin:
        print("[ERROR] No complete results found. Run training first.")
        sys.exit(1)

    # ── Cross-SNR paired test (主检验) ──────────────────────────────────────
    delta_nk_all = np.concatenate(all_deltas_nokin)   # (n_complete × 5,)
    delta_k_all  = np.concatenate(all_deltas_kin)

    print("=" * 80)
    print(f"  Cross-SNR paired test (main test, D26)")
    print(f"  N = {len(delta_nk_all)} pairs  ({len(all_deltas_nokin)} SNR pts × 5 seeds)")
    print("=" * 80)

    for label, deltas in [("Dual_nokin vs Single_nokin", delta_nk_all),
                           ("Dual_kin   vs Single_kin",   delta_k_all)]:
        mean_d = deltas.mean() * 100
        se_d   = deltas.std(ddof=1) / np.sqrt(len(deltas)) * 100
        # Wilcoxon (non-parametric)
        try:
            pw = wilcoxon(deltas, alternative="greater").pvalue
        except Exception:
            pw = float("nan")
        # One-sample t-test (H0: mean=0, H1: mean>0)
        try:
            pt = ttest_1samp(deltas, 0, alternative="greater").pvalue
        except Exception:
            pt = float("nan")
        print(f"  {label}")
        print(f"    mean Δ = {mean_d:+.3f}pp ± {se_d:.3f}pp (SE)")
        print(f"    Wilcoxon p = {pw:.4f}   t-test p = {pt:.4f}")
        frac_pos = (deltas > 0).mean()
        print(f"    positive pairs: {(deltas > 0).sum()}/{len(deltas)} ({frac_pos*100:.0f}%)")
        print()

    # ── Per-SNR Wilcoxon (辅助表) ─────────────────────────────────────────
    print("=" * 80)
    print("  Per-SNR Wilcoxon (n=5 seeds, auxiliary)")
    print("=" * 80)
    print(f"  {'SNR':>5}  {'p_nokin':>10}  {'p_kin':>10}  Δ_nokin   Δ_kin    Levene_nokin")
    for snr, (s_nk, d_nk, s_k, d_k, dn_nk, dn_k) in sorted(per_snr.items()):
        try:
            pn = wilcoxon(d_nk, s_nk, alternative="greater").pvalue
        except Exception:
            pn = float("nan")
        try:
            pk = wilcoxon(d_k, s_k, alternative="greater").pvalue
        except Exception:
            pk = float("nan")
        try:
            pl = levene(d_nk, s_nk).pvalue
        except Exception:
            pl = float("nan")
        print(f"  {snr:3d}dB  p={pn:.3f}      p={pk:.3f}      "
              f"{dn_nk.mean()*100:+.2f}pp  {dn_k.mean()*100:+.2f}pp  Levene_p={pl:.3f}")
    print()

    # ── Δmean vs SNR trend ──────────────────────────────────────────────────
    print("=" * 80)
    print("  Δmean vs SNR (for Discussion narrative)")
    print("=" * 80)
    snrs_done = sorted(per_snr.keys())
    print(f"  {'SNR':>5}  Δ_nokin    Δ_kin     (expected: larger gain at lower SNR)")
    for snr in snrs_done:
        _, _, _, _, dn_nk, dn_k = per_snr[snr]
        print(f"  {snr:3d}dB  {dn_nk.mean()*100:+.3f}pp  {dn_k.mean()*100:+.3f}pp")

    # ── Mixed-effects (optional) ────────────────────────────────────────────
    print()
    try:
        import pingouin as pg
        import pandas as pd

        rows = []
        for snr, (s_nk, d_nk, s_k, d_k, _, _) in per_snr.items():
            for seed_i, (s, d) in enumerate(zip(s_nk, d_nk)):
                rows.append({"snr": snr, "seed": seed_i,
                             "sensor": "single", "acc": float(s), "kind": "nokin"})
                rows.append({"snr": snr, "seed": seed_i,
                             "sensor": "dual",   "acc": float(d), "kind": "nokin"})
        df = pd.DataFrame(rows)
        lm = pg.mixed_anova(data=df, dv="acc", within="sensor",
                            between="snr", subject="seed")
        print("=" * 80)
        print("  Mixed ANOVA (sensor within, snr between, seed as subject)")
        print("=" * 80)
        print(lm[["Source", "F", "p-unc", "np2"]].to_string(index=False))
        print()
    except ImportError:
        print("  [INFO] pingouin not installed — mixed-effects skipped")
        print("         install with: pip install pingouin pandas")
    except Exception as e:
        print(f"  [WARN] Mixed-effects failed: {e}")

    # ── Save JSON summary ────────────────────────────────────────────────────
    out = {}
    for snr, (s_nk, d_nk, s_k, d_k, dn_nk, dn_k) in per_snr.items():
        out[str(snr)] = {
            "single_nokin": s_nk.tolist(),
            "dual_nokin":   d_nk.tolist(),
            "single_kin":   s_k.tolist(),
            "dual_kin":     d_k.tolist(),
            "delta_nokin":  dn_nk.tolist(),
            "delta_kin":    dn_k.tolist(),
        }
    save_path = ROOT / "results" / "b2_snr_curve_summary.json"
    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Summary saved to {save_path}")


if __name__ == "__main__":
    main()
