#!/usr/bin/env python3
"""
experiments/exp_b2_dual_sensor/run_b2_snr_curve.py

B2 双传感器 SNR 曲线实验队列（D26 扩展版）
EAAI 网格 {-8,-6,-4,-2,0} dB × {Dual_nokin, Dual_kin} × 5 seeds

幂等：summary.json 存在则跳过该 run。
SNR=0dB 已完成（exp_b2_dual_{nokin,kin}）自动跳过。

用法:
  cd ~/论文8
  source venv/bin/activate
  python experiments/exp_b2_dual_sensor/run_b2_snr_curve.py [--smoke] [--analyze-only]
"""
import argparse
import subprocess
import sys
import textwrap
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent.parent   # ~/论文8

# ── EAAI SNR 网格 ─────────────────────────────────────────────────────────────
# 0dB 已在 exp_b2_dual_{nokin,kin} 完成；这里覆盖完整网格以便幂等检查
SNR_GRID = [-8, -6, -4, -2, 0]

TRAIN_SCRIPT = ROOT / "experiments/exp01_cwru_baseline/train.py"


def snr_tag(snr: float) -> str:
    return f"snrm{abs(int(snr))}" if snr < 0 else f"snr{int(snr)}"


def results_dir(kind: str, snr: float) -> Path:
    """kind = 'nokin' or 'kin'"""
    if snr == 0:
        # 复用已完成的 0dB 结果
        return (ROOT / f"results/exp_b2_dual_{kind}").expanduser()
    return (ROOT / f"results/exp_b2_dual_{kind}_{snr_tag(snr)}").expanduser()


def make_config(kind: str, snr: float) -> dict:
    lk = 0.0 if kind == "nokin" else 0.01
    tag = snr_tag(snr)
    name = f"exp_b2_dual_{kind}_{tag}" if snr != 0 else f"exp_b2_dual_{kind}"
    cfg = {
        "name": name,
        "data_dir": "~/论文8/data/cwru_12k_de",
        "channels": ["DE", "FE"],
        "win_len": 2048,
        "stride": 1024,
        "val_ratio": 0.2,
        "batch_size": 64,
        "num_workers": 4,
        "d_model": 64,
        "d_state": 128,
        "n_layers": 4,
        "n_classes": 4,
        "conv_stride": 2,
        "epochs": 50,
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "grad_clip": 1.0,
        "scheduler": "cosine",
        "noise_snr_db": float(snr),
        "lambda_kin": lk,
        "kin_variant": "cover",
        # L_kin target: DE-end bearing (fault is at drive-end, SKF 6205)
        # FE bearing (CWRU 6203, D=28.499mm) ≠ PU 6203 (D=29.05mm) — D11
        **({"bearing_kwargs": {"n_balls": 9, "d": 7.938, "D": 39.040}} if lk > 0 else {}),
        "seeds": [0, 1, 2, 3, 4],
        "results_dir": str(results_dir(kind, snr)),
    }
    return cfg


def run_experiment(kind: str, snr: float, smoke: bool) -> bool:
    """Run one experiment. Returns True if ran, False if skipped."""
    rdir = results_dir(kind, snr)
    summary = rdir / "summary.json"
    if summary.exists():
        print(f"  [SKIP] {rdir.name} — summary.json exists")
        return False

    cfg = make_config(kind, snr)
    rdir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        cfg_path = f.name

    cmd = [sys.executable, str(TRAIN_SCRIPT), "--config", cfg_path]
    if smoke:
        cmd += ["--smoke", "--seeds", "0"]
    print(f"\n  [RUN ] {rdir.name}  SNR={snr}dB  lambda_kin={cfg['lambda_kin']}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  [ERR ] exit {result.returncode}")
        sys.exit(result.returncode)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2-epoch / seed-0 smoke test for each config")
    ap.add_argument("--analyze-only", action="store_true",
                    help="Skip training, only run analysis")
    args = ap.parse_args()

    if not args.analyze_only:
        print("=" * 65)
        print("  B2 Dual-Sensor SNR Curve  (D26 extended)")
        print(f"  Grid: {SNR_GRID} dB  ×  {{nokin, kin}}  ×  5 seeds")
        print("=" * 65)

        for snr in SNR_GRID:
            for kind in ("nokin", "kin"):
                run_experiment(kind, snr, smoke=args.smoke)

    if not args.smoke:
        print("\n" + "=" * 65)
        print("  B2 Analysis: cross-SNR paired test")
        print("=" * 65)
        # Import analysis inline so script can run even if analysis deps missing
        analysis_script = Path(__file__).parent / "analyze_b2.py"
        if analysis_script.exists():
            result = subprocess.run([sys.executable, str(analysis_script)], cwd=str(ROOT))
            if result.returncode != 0:
                print("[WARN] Analysis failed — check analyze_b2.py")
        else:
            print("[WARN] analyze_b2.py not found — run separately")


if __name__ == "__main__":
    main()
