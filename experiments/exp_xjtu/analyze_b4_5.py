#!/usr/bin/env python3
"""
B4-5 analysis: single vs dual-channel comparison (D28 protocol).

Primary metric: LOBO macro_recall (Wilcoxon n=5)
Secondary:      per-class (OR, IR) recall, Cross macro_F1 / IR_recall

C3 success criterion (D28):
  dual_channel shows consistent positive gain in both CWRU (B2, -4dB p=0.031)
  and XJTU (B4-5 LOBO, Wilcoxon n=5). If p<0.05 in B4-5 → "both statistically significant".
"""
import json
import pathlib
import numpy as np
from scipy import stats

ROOT = pathlib.Path("/home/jeffwork/论文8/results")

CONFIGS = {
    "lobo_single_nokin":  ROOT / "exp_xjtu_lobo_nokin",
    "lobo_single_kin":    ROOT / "exp_xjtu_lobo_kin",
    "lobo_dual_nokin":    ROOT / "exp_xjtu_lobo_dual_nokin",
    "lobo_dual_kin":      ROOT / "exp_xjtu_lobo_dual_kin",
    "cross_single_nokin": ROOT / "exp_xjtu_cross_nokin",
    "cross_single_kin":   ROOT / "exp_xjtu_cross_kin",
    "cross_dual_nokin":   ROOT / "exp_xjtu_cross_dual_nokin",
    "cross_dual_kin":     ROOT / "exp_xjtu_cross_dual_kin",
}


def load_summary(path: pathlib.Path) -> dict | None:
    p = path / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def wilcoxon_p(a, b) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    try:
        _, p = stats.wilcoxon(a, b)
        return float(p)
    except Exception:
        return float("nan")


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    data = {k: load_summary(v) for k, v in CONFIGS.items()}

    # ── LOBO results ──────────────────────────────────────────────
    print_section("LOBO Cond3 macro_recall (primary metric)")
    lobo_keys = [
        ("lobo_single_nokin", "Single CE-only"),
        ("lobo_single_kin",   "Single +L_kin "),
        ("lobo_dual_nokin",   "Dual   CE-only"),
        ("lobo_dual_kin",     "Dual   +L_kin "),
    ]
    lobo_vals = {}
    print(f"{'Config':22s}  mean±std       per-seed")
    for k, label in lobo_keys:
        d = data.get(k)
        if d is None:
            print(f"{label:22s}  [MISSING]")
            lobo_vals[k] = None
            continue
        seeds = d["per_seed_macro_recall"]
        mean = np.mean(seeds) * 100
        std  = np.std(seeds, ddof=1) * 100 if len(seeds) > 1 else 0.0
        print(f"{label:22s}  {mean:.2f}±{std:.2f}%  {[f'{v*100:.2f}' for v in seeds]}")
        lobo_vals[k] = seeds

    # LOBO statistical tests: dual vs single (primary C3 evidence)
    print()
    print("LOBO Wilcoxon n=5 (dual vs single):")
    for kind in ("nokin", "kin"):
        sng = lobo_vals.get(f"lobo_single_{kind}")
        dbl = lobo_vals.get(f"lobo_dual_{kind}")
        if sng and dbl:
            p = wilcoxon_p(sng, dbl)
            delta = (np.mean(dbl) - np.mean(sng)) * 100
            sig = "✓ SIGNIFICANT" if p < 0.05 else "(not significant)"
            print(f"  {kind:6s}: Δ={delta:+.2f}pp  p={p:.4f}  {sig}")

    # ── Cross-condition results ────────────────────────────────────
    print_section("Cross-condition Cond2→Cond3 macro_F1 + IR_recall (mechanism evidence)")
    cross_keys = [
        ("cross_single_nokin", "Single CE-only"),
        ("cross_single_kin",   "Single +L_kin "),
        ("cross_dual_nokin",   "Dual   CE-only"),
        ("cross_dual_kin",     "Dual   +L_kin "),
    ]
    cross_f1_vals = {}
    print(f"{'Config':22s}  macro_F1       per-seed_F1  |  macro_recall")
    for k, label in cross_keys:
        d = data.get(k)
        if d is None:
            print(f"{label:22s}  [MISSING]")
            cross_f1_vals[k] = None
            continue
        f1s  = d["macro_f1s"]
        recs = d.get("macro_recalls", [])
        mf1  = np.mean(f1s) * 100
        sf1  = np.std(f1s, ddof=1) * 100 if len(f1s) > 1 else 0.0
        mrec = np.mean(recs) * 100 if recs else float("nan")
        print(f"{label:22s}  {mf1:.2f}±{sf1:.2f}%  {[f'{v*100:.2f}' for v in f1s]}  |  {mrec:.2f}%")
        cross_f1_vals[k] = f1s

    print()
    print("Cross Wilcoxon n=5 (dual vs single — mechanism evidence only):")
    for kind in ("nokin", "kin"):
        sng = cross_f1_vals.get(f"cross_single_{kind}")
        dbl = cross_f1_vals.get(f"cross_dual_{kind}")
        if sng and dbl:
            p = wilcoxon_p(sng, dbl)
            delta = (np.mean(dbl) - np.mean(sng)) * 100
            sig = "✓ SIGNIFICANT" if p < 0.05 else "(not significant)"
            print(f"  {kind:6s}: Δ={delta:+.2f}pp  p={p:.4f}  {sig}")

    # ── C3 evidence chain summary ──────────────────────────────────
    print_section("C3 Evidence Chain (D28 criterion)")
    print("CWRU B2:  Dual vs Single, -4dB nokin, Wilcoxon p=0.031 ✓ (already established)")
    sng_n = lobo_vals.get("lobo_single_nokin")
    dbl_n = lobo_vals.get("lobo_dual_nokin")
    if sng_n and dbl_n:
        p_lobo = wilcoxon_p(sng_n, dbl_n)
        delta  = (np.mean(dbl_n) - np.mean(sng_n)) * 100
        if p_lobo < 0.05:
            verdict = "✓ SIGNIFICANT — C3 达'两处统计显著'最强等级"
        elif delta > 0:
            verdict = "方向一致正向增益，但不显著 — C3 仍可叙述'两数据集方向一致'"
        else:
            verdict = "⚠️ 负向增益 — C3 叙事需重新评估"
        print(f"XJTU B4-5: LOBO dual vs single nokin: Δ={delta:+.2f}pp  p={p_lobo:.4f}  {verdict}")

    # ── Save JSON ──────────────────────────────────────────────────
    out = {}
    for k, label in lobo_keys + cross_keys:
        d = data.get(k)
        out[k] = {"missing": True} if d is None else d
    for kind in ("nokin", "kin"):
        sng = lobo_vals.get(f"lobo_single_{kind}")
        dbl = lobo_vals.get(f"lobo_dual_{kind}")
        if sng and dbl:
            p = wilcoxon_p(sng, dbl)
            delta = (np.mean(dbl) - np.mean(sng)) * 100
            out[f"lobo_wilcoxon_{kind}"] = {"delta_pp": delta, "p": p}
    (ROOT / "b4_5_summary.json").write_text(json.dumps(out, indent=2))
    print(f"\n→ Saved results/b4_5_summary.json")


if __name__ == "__main__":
    main()
