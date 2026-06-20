r"""
tools/cwru_inventory.py — CWRU 数据盘点、软链接创建、缺失文件下载

用法:
  python tools/cwru_inventory.py [--source-dir DIR] [--download]

步骤:
  1. 扫描 source_dirs 下全部 .mat，提取真实 file_id（X(\d+)_DE_time）
  2. 对照 MANIFEST 40 个 ID：命中的建软链接到 data/cwru_12k_de/{id}.mat
  3. 报告命中/缺失
  4. --download: 对缺失 ID 尝试从 CWRU 官网下载
  5. 全量复核：读取 40 个链接，验证 DE 通道和 RPM 变量存在
"""
import re
import sys
import argparse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from bearmamba3.data_cwru import MANIFEST, _load_mat, read_file_id

LINK_DIR = PROJECT_ROOT / "data" / "cwru_12k_de"

# ── CWRU 官方下载 URL 模板（2024 年有效，如失效见下方备用）──────────────────
# 官方: https://engineering.case.edu/bearingdatacenter/download-data-file
# 直链: https://engineering.case.edu/sites/default/files/bearing_data_center/
#       12k_Drive_End_Bearing_Fault_Data/{id}.mat  （部分 ID 在此路径）
# 备用镜像: https://github.com/XiaobuLv0626/CWRU-Bearing-Dataset （仅作参考）
CWRU_URL_TEMPLATES = [
    "https://engineering.case.edu/sites/default/files/{id}.mat",          # 2024+ 官网直链
    "https://engineering.case.edu/sites/default/files/bearing_data_center/12k_Drive_End_Bearing_Fault_Data/{id}.mat",
    "https://engineering.case.edu/sites/default/files/bearing_data_center/Normal_Baseline_Data/{id}.mat",
]

SOURCE_DIRS_DEFAULT = [
    Path.home() / "data_cwru",
    Path.home() / "data",
    Path.home() / "mamba_test" / "data",
]


def scan_source_dirs(source_dirs: list[Path]) -> dict[int, Path]:
    """返回 {file_id: mat_path} — 用文件内变量名识别，不信任文件名。
    也包含 LINK_DIR 中已直接存在（非软链接）的文件，
    以便直接下载到目标目录的文件也被计入。
    """
    found: dict[int, Path] = {}
    all_dirs = list(source_dirs) + [LINK_DIR]
    for d in all_dirs:
        if not d.exists():
            continue
        for mat in sorted(d.glob("*.mat")):
            # 跳过软链接（链接已有源，不重复计）
            if mat.is_symlink():
                continue
            fid = read_file_id(mat)
            if fid is not None and fid in MANIFEST and fid not in found:
                found[fid] = mat
    # 第二轮：源目录中的文件（覆盖上面可能的直接文件）
    for d in source_dirs:
        if not d.exists():
            continue
        for mat in sorted(d.glob("*.mat")):
            fid = read_file_id(mat)
            if fid is not None and fid in MANIFEST:
                found[fid] = mat  # 源目录文件优先（会建软链接）
    return found


def make_symlinks(found: dict[int, Path]):
    LINK_DIR.mkdir(parents=True, exist_ok=True)
    for fid, src in found.items():
        link = LINK_DIR / f"{fid}.mat"
        # 源文件已在目标目录中（直接下载），不需要建软链接
        if src.resolve() == link.resolve():
            continue
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            continue  # 真实文件已存在，不覆盖
        link.symlink_to(src.resolve())


def try_download(fid: int, dest: Path) -> bool:
    for tmpl in CWRU_URL_TEMPLATES:
        url = tmpl.format(id=fid)
        try:
            print(f"  → 尝试下载 {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) < 1000:  # 太小，不是真实 .mat
                continue
            dest.write_bytes(data)
            fid_check = read_file_id(dest)
            if fid_check == fid:
                print(f"  ✅ {fid}.mat 下载成功 ({len(data)//1024} kB)")
                return True
            else:
                dest.unlink(missing_ok=True)
        except Exception as e:
            print(f"  ✗ {url} 失败: {e}")
    return False


def verify_all() -> list[str]:
    errors = []
    manifest_ids = sorted(MANIFEST.keys())
    print(f"\n{'─'*60}")
    print(f"{'ID':>4}  {'链接存在':^8}  {'loadmat':^8}  {'DE通道':^8}  {'RPM变量':^8}  {'真实ID':^6}")
    print(f"{'─'*60}")
    for fid in manifest_ids:
        link = LINK_DIR / f"{fid}.mat"
        link_ok = link.exists()
        loadmat_ok = de_ok = rpm_ok = real_id_ok = False
        real_id = "—"
        if link_ok:
            try:
                data = _load_mat(link)
                loadmat_ok = True
                de_key = f"X{fid:03d}_DE_time"
                rpm_key = f"X{fid:03d}RPM"
                de_ok = de_key in data
                # RPM missing in some CWRU files (e.g. 98) — fall back to nominal, not an error
                rpm_ok = rpm_key in data
                rid = read_file_id(link)
                real_id = str(rid) if rid else "?"
                real_id_ok = (rid == fid)
            except Exception as e:
                errors.append(f"{fid}: loadmat error — {e}")
        sym = "✅" if link_ok else "❌"
        lm  = "✅" if loadmat_ok else ("❌" if link_ok else "—")
        de  = "✅" if de_ok  else ("❌" if loadmat_ok else "—")
        rp  = ("✅" if rpm_ok else "⚠nom") if loadmat_ok else "—"  # ⚠nom = will use nominal
        ri  = ("✅" if real_id_ok else f"⚠{real_id}") if loadmat_ok else "—"
        print(f"{fid:>4}  {sym:^8}  {lm:^8}  {de:^8}  {rp:^8}  {ri:^6}")
        # RPM missing is a warning not an error; only flag real failures
        if link_ok and not (loadmat_ok and de_ok and real_id_ok):
            errors.append(f"{fid}: 验证未全通过 (loadmat={loadmat_ok} de={de_ok} id={real_id_ok})")
    print(f"{'─'*60}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", nargs="*", type=Path,
                        help="额外搜索目录（默认搜索 ~/data_cwru ~/data）")
    parser.add_argument("--download", action="store_true",
                        help="对缺失 ID 尝试从 CWRU 官网下载")
    args = parser.parse_args()

    source_dirs = (args.source_dir or []) + SOURCE_DIRS_DEFAULT

    print("=" * 60)
    print("CWRU 数据盘点")
    print("=" * 60)

    # ── Step 1: 扫描 ──────────────────────────────────────────────
    print(f"\n[1] 扫描源目录: {[str(d) for d in source_dirs if d.exists()]}")
    found = scan_source_dirs(source_dirs)
    manifest_ids = set(MANIFEST.keys())
    hit_ids  = sorted(found.keys())
    miss_ids = sorted(manifest_ids - set(found.keys()))

    print(f"    命中: {len(hit_ids)}/40  → {hit_ids}")
    print(f"    缺失: {len(miss_ids)}/40  → {miss_ids}")

    # ── Step 2: 建软链接 ──────────────────────────────────────────
    print(f"\n[2] 建软链接 → {LINK_DIR}")
    make_symlinks(found)
    print(f"    已建链接: {len(hit_ids)} 个")

    # ── Step 3: 下载缺失 ──────────────────────────────────────────
    if miss_ids:
        if args.download:
            print(f"\n[3] 尝试下载 {len(miss_ids)} 个缺失文件...")
            still_missing = []
            for fid in miss_ids:
                dest = LINK_DIR / f"{fid}.mat"
                ok = try_download(fid, dest)
                if not ok:
                    still_missing.append(fid)
            if still_missing:
                print(f"\n    ⚠️  仍缺失 {len(still_missing)} 个: {still_missing}")
                print("    请手动从 https://engineering.case.edu/bearingdatacenter/download-data-file")
                print("    下载后放入任意源目录，再不带 --download 重跑本脚本。")
        else:
            print(f"\n[3] 缺失 {len(miss_ids)} 个 ID，补充方法:")
            print(f"    a) 重跑加 --download 自动尝试从 CWRU 官网下载")
            print(f"    b) 手动下载后放入 {LINK_DIR.parent.parent}/data_cwru/ 再重跑")
            print(f"    缺失 ID: {miss_ids}")

    # ── Step 4: 全量复核 ──────────────────────────────────────────
    print(f"\n[4] 全量复核（{len(hit_ids)} 个已有文件）:")
    errors = verify_all()

    print(f"\n{'='*60}")
    if errors:
        print(f"❌ 发现 {len(errors)} 处问题:")
        for e in errors:
            print(f"   {e}")
    else:
        n_ready = sum(1 for fid in MANIFEST if (LINK_DIR / f"{fid}.mat").exists())
        print(f"✅ {n_ready}/40 个文件验证通过" +
              (f"，缺 {40-n_ready} 个待补下载" if n_ready < 40 else "，数据集完整！"))


if __name__ == "__main__":
    main()
