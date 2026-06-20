"""
exp07: SK-SVM baseline — Spectral Kurtosis features + SVM

特征集（10 维）:
  RMS, kurtosis, skewness, crest factor, peak-to-peak,      # time domain
  FFT top-k energy ratio (k=5 bins), spectral centroid,
  spectral kurtosis (4th spectral moment)                    # frequency domain

用法:
  python experiments/exp07_baselines/train_sksvm.py [--snr_db -4] [--seed 0]
  python experiments/exp07_baselines/train_sksvm.py --run_all    # 全 SNR 网格 × 5 seeds
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.signal import hilbert
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bearmamba3.data_cwru import CWRUDataset

SNR_GRID = [None, 0.0, -2.0, -4.0, -6.0, -8.0]


def extract_features(x: np.ndarray) -> np.ndarray:
    """x: (win_len,) — single-channel signal → feature vector (12,)"""
    # Time domain
    rms        = np.sqrt(np.mean(x**2))
    kurt       = float(stats.kurtosis(x, fisher=True))
    skew       = float(stats.skew(x))
    crest      = np.max(np.abs(x)) / (rms + 1e-8)
    pp         = np.ptp(x)

    # Frequency domain via FFT
    N     = len(x)
    mag   = np.abs(np.fft.rfft(x)) / N
    freqs = np.arange(len(mag))

    # Top-5 energy ratio
    top5_energy = np.sort(mag**2)[-5:].sum() / (np.sum(mag**2) + 1e-8)

    # Spectral centroid
    spec_cent = np.sum(freqs * mag) / (np.sum(mag) + 1e-8)

    # Spectral kurtosis (4th normalized spectral moment)
    mu   = spec_cent
    sig2 = np.sum(mag * (freqs - mu)**2) / (np.sum(mag) + 1e-8)
    sig4 = np.sum(mag * (freqs - mu)**4) / (np.sum(mag) + 1e-8)
    spec_kurt = sig4 / (sig2**2 + 1e-8) - 3.0

    # Envelope spectrum: kurtosis of Hilbert envelope
    env = np.abs(hilbert(x))
    env_kurt = float(stats.kurtosis(env, fisher=True))

    # Energy ratio in 1st quartile vs 4th quartile of spectrum
    q1 = mag[:N//8].sum()
    q4 = mag[-N//8:].sum()
    band_ratio = q1 / (q4 + 1e-8)

    return np.array([rms, kurt, skew, crest, pp,
                     top5_energy, spec_cent, spec_kurt,
                     env_kurt, band_ratio,
                     np.log1p(rms), np.log1p(np.abs(spec_cent))], dtype=np.float32)


def build_dataset(snr_db, seed, win_len=2048, stride=1024, val_ratio=0.2, smoke=False):
    ds = CWRUDataset(
        data_dir=str(Path("~/论文8/data/cwru_12k_de").expanduser()),
        win_len=win_len, stride=stride, channels=["DE"],
        normalize=True, seed=seed, noise_snr_db=snr_db, label_mode="4class",
    )
    n_total = len(ds)
    n_val   = int(n_total * val_ratio)
    n_train = n_total - n_val
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_total)
    if smoke:
        idx = idx[:200]
        n_train = 160; n_val = 40
    train_idx = idx[:n_train]; val_idx = idx[n_train:n_train+n_val]

    def extract_split(indices):
        X, y = [], []
        for i in indices:
            x_t, lbl, _ = ds[i]
            X.append(extract_features(x_t[0].numpy()))   # channel 0
            y.append(int(lbl))
        return np.stack(X), np.array(y)

    print("  Extracting train features ...", flush=True)
    X_tr, y_tr = extract_split(train_idx)
    print("  Extracting val features ...", flush=True)
    X_va, y_va = extract_split(val_idx)
    return X_tr, y_tr, X_va, y_va


def run_one(snr_db, seed, smoke=False):
    X_tr, y_tr, X_va, y_va = build_dataset(snr_db, seed, smoke=smoke)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(kernel="rbf", C=10.0, gamma="scale",
                       decision_function_shape="ovr", random_state=seed)),
    ])
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_va, model.predict(X_va))
    return acc


def snr_tag(snr_db):
    return "clean" if snr_db is None else f"snr{int(snr_db):+d}".replace("+", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr_db",  type=float, default=None)
    ap.add_argument("--seed",    type=int,   default=0)
    ap.add_argument("--run_all", action="store_true")
    ap.add_argument("--smoke",   action="store_true")
    args = ap.parse_args()

    results_root = Path("~/论文8/results").expanduser()

    if args.run_all:
        for snr in SNR_GRID:
            tag   = snr_tag(snr)
            seeds = [0, 1, 2, 3, 4]
            accs  = []
            for s in seeds:
                print(f"SK-SVM | {tag} | seed={s}", flush=True)
                acc = run_one(snr, s, smoke=args.smoke)
                accs.append(acc)
                print(f"  acc={acc*100:.2f}%")
            mean = float(np.mean(accs)); std = float(np.std(accs, ddof=1))
            print(f"  {tag}: mean={mean*100:.2f}±{std*100:.2f}%")
            out_dir = results_root / f"exp07_sksvm_{tag}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "metrics.json", "w") as f:
                json.dump({"mean_acc": mean, "std_acc": std,
                           "per_seed": accs, "snr_db": snr,
                           "backbone": "sksvm", "n_seeds": 5}, f, indent=2)
            print(f"  Saved → {out_dir}/metrics.json")
    else:
        snr = args.snr_db
        tag = snr_tag(snr)
        print(f"SK-SVM | {tag} | seed={args.seed}", flush=True)
        acc = run_one(snr, args.seed, smoke=args.smoke)
        print(f"  acc={acc*100:.2f}%")
        out_dir = results_root / f"exp07_sksvm_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"seed{args.seed}.json"
        with open(fname, "w") as f:
            json.dump({"acc": acc, "snr_db": snr, "seed": args.seed}, f, indent=2)
        print(f"  Saved → {fname}")


if __name__ == "__main__":
    main()
