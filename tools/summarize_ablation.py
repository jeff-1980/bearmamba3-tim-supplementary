"""
tools/summarize_ablation.py — print mean±std for a results directory

Usage:
  python tools/summarize_ablation.py results/exp02_snr-5_nokin/
  python tools/summarize_ablation.py results/  # all subdirs
"""
import json
import sys
from pathlib import Path

import numpy as np


def summarize(results_dir: Path):
    files = sorted(results_dir.glob("seed_*.json"))
    if not files:
        return None
    accs = []
    for f in files:
        d = json.loads(f.read_text())
        accs.append(d["best_val_acc"])
    return {
        "n":    len(accs),
        "mean": float(np.mean(accs)),
        "std":  float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "accs": accs,
    }


def main():
    roots = [Path(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [Path("results")]

    for root in roots:
        if (root / "seed_0.json").exists():
            targets = [root]
        else:
            targets = sorted(d for d in root.iterdir() if d.is_dir())

        for d in targets:
            r = summarize(d)
            if r is None:
                continue
            accs_str = "/".join(f"{a*100:.2f}" for a in r["accs"])
            print(f"{d.name:<35}  n={r['n']}  "
                  f"{r['mean']*100:.2f}±{r['std']*100:.2f}%  [{accs_str}]")


if __name__ == "__main__":
    main()
