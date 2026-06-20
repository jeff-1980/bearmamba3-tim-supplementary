"""
bearmamba3/data_pu.py — Paderborn University KAt 数据管道

轴承: SKF 6203  Z=8  d=6.75mm  D=29.05mm(pitch)  α=0°  fs=64kHz
  → BEARING_KWARGS 传给 kinematic_loss / compute_fault_freqs

文件命名: {COND}_{BEARING}_{N}.mat
  COND:    N09_M07_F10 | N15_M01_F10 | N15_M07_F04 | N15_M07_F10
  BEARING: K001/K002=Normal  KA04/KA15=Outer  KI01/KI03/KI05=Inner
           KB23/KB24/KB27=Real damage（3类实验中跳过）

3类标签: 0=Normal  1=Outer  2=Inner
返回三元组 (x, label, rpm) — 与 data_cwru.py 接口一致

跨工况默认分组:
  训练: N09_M07_F10 + N15_M01_F10
  测试: N15_M07_F10
"""
import glob
import os

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

# ── 轴承参数 —————————————————————————————————————————————————
FS = 64_000   # Hz（官方值 64,498 Hz，取整 64kHz）

# SKF 6203: Z=8, d=6.75mm, D=29.05mm pitch circle, α=0°
# 传给 compute_fault_freqs / kinematic_loss 时使用（单位 mm，比值相同）
BEARING_KWARGS = {"n_balls": 8, "d": 6.75, "D": 29.05, "contact_angle_deg": 0.0}

# ── 标签映射 —————————————————————————————————————————————————
LABEL_MAP = {
    "K001": 0, "K002": 0,
    "KA04": 1, "KA15": 1,
    "KI01": 2, "KI03": 2, "KI05": 2,
}
LABEL_NAMES = {0: "Normal", 1: "Outer", 2: "Inner"}

# 工况 → 转速 RPM
COND_RPM = {
    "N09_M07_F10": 900,
    "N15_M01_F10": 1500,
    "N15_M07_F04": 1500,
    "N15_M07_F10": 1500,
}

COND_TRAIN = {"N09_M07_F10", "N15_M01_F10"}
COND_TEST  = {"N15_M07_F10"}


def _parse_fname(path: str):
    """'.../{COND}_{BEARING}_{N}.mat' → (cond_str, bearing_key)"""
    base = os.path.basename(path).replace(".mat", "")
    parts = base.split("_")
    # Format: N09_M07_F10_KI01_3  → parts[0-2]=cond, parts[3]=bearing
    if len(parts) < 5:
        return None, None
    cond    = "_".join(parts[:3])
    bearing = parts[3]
    return cond, bearing


def _load_vib(path: str) -> np.ndarray:
    """Load vibration signal (channel 1) from a PU .mat file."""
    key = os.path.basename(path).replace(".mat", "")
    mat = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    vib = np.array(mat[key].Y[6].Data).squeeze().astype(np.float32)
    return vib


def _sliding_windows(arr: np.ndarray, win: int, stride: int) -> np.ndarray:
    n = (len(arr) - win) // stride + 1
    idx = np.arange(win)[None, :] + stride * np.arange(n)[:, None]
    return arr[idx]    # (n, win)


class PUDataset(Dataset):
    """
    Paderborn University KAt dataset, compatible with BearMamba3 training loop.

    Returns (x, label, rpm) where:
      x     : (1, win_len) float32 — single vibration channel (SISO)
      label : int64 scalar — 0=Normal, 1=Outer, 2=Inner
      rpm   : float32 scalar — shaft speed from condition code (rpm)

    Args:
        data_dir      : path containing {COND}_{BEARING}_{N}.mat files
        conditions    : set of condition codes to load; None = all 4
        win_len       : samples per window (default 4096 ≈ 64ms @64kHz)
        stride        : hop size (default = win_len, non-overlapping)
        normalize     : per-window z-score normalization
        noise_snr_db  : add Gaussian noise at given SNR before normalization
        seed          : not used for data loading (determinism via noise hash)
    """

    def __init__(
        self,
        data_dir:     str,
        conditions:   set  | None = None,
        win_len:      int         = 4096,
        stride:       int  | None = None,
        normalize:    bool        = True,
        noise_snr_db: float| None = None,
        seed:         int         = 0,
    ):
        stride = stride or win_len
        self.win_len       = win_len
        self.normalize     = normalize
        self.noise_snr_db  = noise_snr_db

        all_files = sorted(glob.glob(os.path.join(data_dir, "*.mat")))
        segs, labels, rpms = [], [], []

        for path in all_files:
            cond, bearing = _parse_fname(path)
            if bearing not in LABEL_MAP:
                continue
            if conditions is not None and cond not in conditions:
                continue

            label = LABEL_MAP[bearing]
            rpm   = float(COND_RPM.get(cond, 1500))
            vib   = _load_vib(path)
            wins  = _sliding_windows(vib, win_len, stride)   # (n, win_len)
            n     = len(wins)

            segs.append(wins)
            labels.append(np.full(n, label, dtype=np.int64))
            rpms.append(np.full(n, rpm,   dtype=np.float32))

        self._data   = np.concatenate(segs,   axis=0)   # (N, win_len)
        self._labels = np.concatenate(labels, axis=0)   # (N,)
        self._rpms   = np.concatenate(rpms,   axis=0)   # (N,)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int):
        w = self._data[idx].copy()   # (win_len,)

        if self.noise_snr_db is not None:
            rng = np.random.default_rng(hash((idx, self.noise_snr_db)) & 0xFFFFFFFF)
            sig_pwr = np.mean(w ** 2, keepdims=True).clip(min=1e-12)
            noise_std = np.sqrt(sig_pwr / (10 ** (self.noise_snr_db / 10.0)))
            w = w + rng.standard_normal(w.shape).astype(np.float32) * noise_std

        if self.normalize:
            mu  = w.mean()
            std = w.std() + 1e-8
            w   = (w - mu) / std

        x   = torch.from_numpy(w[None, :])                           # (1, win_len)
        lbl = torch.tensor(self._labels[idx], dtype=torch.long)
        rpm = torch.tensor(self._rpms[idx],   dtype=torch.float32)
        return x, lbl, rpm
