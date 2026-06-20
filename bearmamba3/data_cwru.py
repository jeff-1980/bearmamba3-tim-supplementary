"""
bearmamba3/data_cwru.py — CWRU 12kHz 驱动端数据管道

MANIFEST: 40 个标准文件 ID（4 类 × 10 个条件）
  Normal(0)       : 97-100
  Inner Race(1)   : 105-108, 169-172, 209-212
  Ball(2)         : 118-121, 185-188, 222-225
  Outer Race @6(3): 130-133, 197-200, 234-237

变量命名规律（已由 verify_mamba3 侦察确认）:
  X{id:03d}_DE_time  驱动端加速度
  X{id:03d}_FE_time  风扇端加速度（全部 40 个文件均有）
  X{id:03d}RPM       转速（标量）

数据目录: data/cwru_12k_de/{id}.mat  → 软链接到原始文件
"""
import re
import numpy as np
import scipy.io
import torch
from pathlib import Path
from torch.utils.data import Dataset

# ─── MANIFEST ────────────────────────────────────────────────────────────────
# id -> (fault_type, fault_size_inch, load_hp, nominal_rpm, label)
MANIFEST = {
    # Normal
     97: ("normal", 0.000, 0, 1797, 0),
     98: ("normal", 0.000, 1, 1772, 0),
     99: ("normal", 0.000, 2, 1750, 0),
    100: ("normal", 0.000, 3, 1730, 0),
    # Inner Race 0.007"
    105: ("inner",  0.007, 0, 1797, 1),
    106: ("inner",  0.007, 1, 1772, 1),
    107: ("inner",  0.007, 2, 1750, 1),
    108: ("inner",  0.007, 3, 1730, 1),
    # Ball 0.007"
    118: ("ball",   0.007, 0, 1797, 2),
    119: ("ball",   0.007, 1, 1772, 2),
    120: ("ball",   0.007, 2, 1750, 2),
    121: ("ball",   0.007, 3, 1730, 2),
    # Outer Race @6 0.007"
    130: ("outer",  0.007, 0, 1797, 3),
    131: ("outer",  0.007, 1, 1772, 3),
    132: ("outer",  0.007, 2, 1750, 3),
    133: ("outer",  0.007, 3, 1730, 3),
    # Inner Race 0.014"
    169: ("inner",  0.014, 0, 1797, 1),
    170: ("inner",  0.014, 1, 1772, 1),
    171: ("inner",  0.014, 2, 1750, 1),
    172: ("inner",  0.014, 3, 1730, 1),
    # Ball 0.014"
    185: ("ball",   0.014, 0, 1797, 2),
    186: ("ball",   0.014, 1, 1772, 2),
    187: ("ball",   0.014, 2, 1750, 2),
    188: ("ball",   0.014, 3, 1730, 2),
    # Outer Race @6 0.014"
    197: ("outer",  0.014, 0, 1797, 3),
    198: ("outer",  0.014, 1, 1772, 3),
    199: ("outer",  0.014, 2, 1750, 3),
    200: ("outer",  0.014, 3, 1730, 3),
    # Inner Race 0.021"
    209: ("inner",  0.021, 0, 1797, 1),
    210: ("inner",  0.021, 1, 1772, 1),
    211: ("inner",  0.021, 2, 1750, 1),
    212: ("inner",  0.021, 3, 1730, 1),
    # Ball 0.021"
    222: ("ball",   0.021, 0, 1797, 2),
    223: ("ball",   0.021, 1, 1772, 2),
    224: ("ball",   0.021, 2, 1750, 2),
    225: ("ball",   0.021, 3, 1730, 2),
    # Outer Race @6 0.021"
    234: ("outer",  0.021, 0, 1797, 3),
    235: ("outer",  0.021, 1, 1772, 3),
    236: ("outer",  0.021, 2, 1750, 3),
    237: ("outer",  0.021, 3, 1730, 3),
}

LABEL_NAMES = {0: "normal", 1: "inner", 2: "ball", 3: "outer"}
FS = 12_000  # Hz

# ─── Bearing geometry for L_kin fault-frequency computation ───────────────────
# CWRU Drive-End (DE) — SKF 6205 (source: CWRU website + literature倍频核算)
DE_BEARING = dict(n_balls=9, d=7.938, D=39.040)   # d=0.3126", D=1.537" → mm (pitch)
# CWRU Fan-End (FE) — 6203 (source: CWRU website)
# ⚠️  D_pitch=28.499mm (28.5mm) ≠ PU dataset's 6203 D_pitch=29.05mm — NEVER interchange!
FE_BEARING = dict(n_balls=8, d=6.746, D=28.499)   # d=0.2656", D=1.122" → mm (pitch)
# fs_eff = FS / conv_stride (conv_stride=2 → 6000 Hz for both DE and FE at 12kHz)

# ─── 10-class label map (fault_type × fault_size) ─────────────────────────────
_SIZE_IDX  = {0.007: 0, 0.014: 1, 0.021: 2}
_TYPE_OFF  = {"inner": 1, "ball": 4, "outer": 7}
LABEL10: dict[int, int] = {}
for _fid, (_ft, _fs, *_) in MANIFEST.items():
    LABEL10[_fid] = 0 if _ft == "normal" else _TYPE_OFF[_ft] + _SIZE_IDX[_fs]

LABEL10_NAMES = {
    0: "normal",
    1: "IR-7mil", 2: "IR-14mil", 3: "IR-21mil",
    4: "ball-7mil", 5: "ball-14mil", 6: "ball-21mil",
    7: "OR-7mil", 8: "OR-14mil", 9: "OR-21mil",
}


def _load_mat(path: Path):
    """Load .mat and return dict of non-header keys."""
    m = scipy.io.loadmat(str(path))
    return {k: v for k, v in m.items() if not k.startswith("_")}


def read_file_id(path: Path) -> int | None:
    r"""Extract true file_id from X(\d+)_DE_time variable name inside .mat.

    Some CWRU files (e.g. 99.mat) contain variables for multiple IDs.
    Returns the highest ID found that is in MANIFEST, so that 99.mat →
    X099_DE_time takes priority over the incidentally-present X098_DE_time.
    Falls back to highest ID found regardless of MANIFEST if none match.
    """
    try:
        keys = list(_load_mat(path).keys())
    except Exception:
        return None
    ids_found = []
    for k in keys:
        m = re.match(r"X(\d+)_DE_time", k)
        if m:
            ids_found.append(int(m.group(1)))
    if not ids_found:
        return None
    # Prefer highest ID that is in MANIFEST
    manifest_ids = [i for i in ids_found if i in MANIFEST]
    return max(manifest_ids) if manifest_ids else max(ids_found)


def load_signal(path: Path, channels: list[str] = ("DE",)) -> tuple[np.ndarray, float]:
    r"""
    Load waveform from .mat file.
    Returns (signal, rpm) where signal has shape (n_channels, L).
    channels: subset of {"DE", "FE", "BA"} — order preserved.
    rpm: scalar float from X{id}RPM variable; falls back to MANIFEST nominal if absent.
    """
    data = _load_mat(path)
    fid = read_file_id(path)
    if fid is None:
        raise ValueError(f"Cannot parse file_id from {path}")

    arrays = []
    for ch in channels:
        key = f"X{fid:03d}_{ch}_time"
        if key not in data:
            raise KeyError(f"{key} not found in {path.name}; available: {list(data)}")
        arrays.append(data[key].squeeze().astype(np.float32))

    rpm_key = f"X{fid:03d}RPM"
    rpm = float(data[rpm_key].squeeze()) if rpm_key in data else float(
        MANIFEST[fid][3]  # fall back to nominal
    )
    signal = np.stack(arrays, axis=0)  # (C, L)
    return signal, rpm


# ─── Dataset ─────────────────────────────────────────────────────────────────

class CWRUDataset(Dataset):
    """
    Sliding-window dataset over CWRU 12kHz DE data.

    Args:
        data_dir  : path to data/cwru_12k_de/ (contains {id}.mat symlinks)
        ids       : subset of MANIFEST keys to use (default: all 40)
        win_len   : samples per window (default 2048 → ~171ms @12kHz)
        stride    : hop between windows (default win_len // 2)
        channels  : list of sensor channels, e.g. ["DE"] or ["DE", "FE"]
        normalize : per-window z-score
        seed      : RNG seed for reproducible train/val split
    """

    def __init__(
        self,
        data_dir: str | Path,
        ids: list[int] | None = None,
        win_len: int = 2048,
        stride: int | None = None,
        channels: list[str] = ("DE",),
        normalize: bool = True,
        seed: int = 0,
        noise_snr_db: float | None = None,
        label_mode: str = "4class",   # "4class" or "10class"
    ):
        self.data_dir = Path(data_dir)
        self.ids = list(ids or MANIFEST.keys())
        self.win_len = win_len
        self.stride = stride if stride is not None else win_len // 2
        self.channels = list(channels)
        self.normalize = normalize
        self.noise_snr_db = noise_snr_db
        self.label_mode = label_mode

        self.samples: list[tuple[np.ndarray, int, float]] = []  # (window, label, rpm)
        self._build(seed)

    def _build(self, seed: int):
        rng = np.random.default_rng(seed)
        for fid in self.ids:
            path = self.data_dir / f"{fid}.mat"
            if not path.exists():
                raise FileNotFoundError(
                    f"data/cwru_12k_de/{fid}.mat not found — run tools/cwru_inventory.py first"
                )
            signal, rpm = load_signal(path, self.channels)
            label = LABEL10[fid] if self.label_mode == "10class" else MANIFEST[fid][4]
            L = signal.shape[1]
            starts = range(0, L - self.win_len + 1, self.stride)
            for s in starts:
                w = signal[:, s : s + self.win_len].copy()
                self.samples.append((w, label, rpm))
        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        w, label, rpm = self.samples[idx]
        if self.noise_snr_db is not None:
            rng = np.random.default_rng(hash((idx, self.noise_snr_db)) & 0xFFFFFFFF)
            sig_pwr = np.mean(w ** 2, axis=1, keepdims=True).clip(min=1e-12)
            noise_std = np.sqrt(sig_pwr / (10 ** (self.noise_snr_db / 10.0)))
            w = w + (rng.standard_normal(w.shape).astype(np.float32) * noise_std)
        if self.normalize:
            mu, std = w.mean(axis=1, keepdims=True), w.std(axis=1, keepdims=True) + 1e-8
            w = (w - mu) / std
        return (
            torch.from_numpy(w),           # (C, win_len)
            torch.tensor(label, dtype=torch.long),
            torch.tensor(rpm, dtype=torch.float32),
        )
