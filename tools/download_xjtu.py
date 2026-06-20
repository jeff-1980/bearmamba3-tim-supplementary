#!/usr/bin/env python3
"""
tools/download_xjtu.py
下载 XJTU-SY 数据集（MediaFire 6分卷 RAR），解压到 ~/data_xjtu/
支持断点续传 + 16 并发分块（绕过单连接限速）
用法: python3 tools/download_xjtu.py
"""
import urllib.request, re, os, subprocess, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

DEST = os.path.expanduser("~/data_xjtu")
os.makedirs(DEST, exist_ok=True)

PARTS = [
    ("58qg5pjkqq5t26r", "XJTU-SY_Bearing_Datasets.part01.rar", 744488960),
    ("sqbsl8ja9c9e84x", "XJTU-SY_Bearing_Datasets.part02.rar", 744488960),
    ("fj8e74g01vwn5hy", "XJTU-SY_Bearing_Datasets.part03.rar", 744488960),
    ("domu7dirov1wjc8", "XJTU-SY_Bearing_Datasets.part04.rar", 744488960),
    ("n5ozw16n3wqgdaw", "XJTU-SY_Bearing_Datasets.part05.rar", 744488960),
    ("ts2majip39fcnq1", "XJTU-SY_Bearing_Datasets.part06.rar", 722155640),
]
CHUNK_SIZE = 8 * 1024 * 1024   # 8 MB per chunk
PARALLEL   = 8                  # concurrent chunks

def get_direct_url(key, filename):
    page = f"https://www.mediafire.com/file/{key}/{filename}/file"
    req = urllib.request.Request(page, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
    with urllib.request.urlopen(req, timeout=25) as r:
        html = r.read().decode("utf-8", errors="ignore")
    m = re.search(r"(https://download\d+\.mediafire\.com/[^\s\"'<>]+\.rar)", html)
    if m:
        return m.group(1)
    raise RuntimeError(f"Could not find direct URL for {filename}")

def fetch_chunk(url, start, end, retries=3):
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)", "Range": f"bytes={start}-{end}"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                return start, r.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

def download_part(key, filename, expected_size):
    dest = os.path.join(DEST, filename)
    existing = os.path.getsize(dest) if os.path.exists(dest) else 0
    if existing == expected_size:
        print(f"  [SKIP] {filename} ({expected_size/1e9:.2f} GB) already complete")
        return

    print(f"  [GET ] direct URL for {filename}...")
    url = get_direct_url(key, filename)
    print(f"  [DL  ] {filename} ({expected_size/1e9:.2f} GB) via {PARALLEL} threads...")

    # build chunk list (skip already-downloaded bytes)
    chunks = []
    pos = existing
    while pos < expected_size:
        end = min(pos + CHUNK_SIZE - 1, expected_size - 1)
        chunks.append((pos, end))
        pos = end + 1

    # write chunks in order using a temp buffer
    lock = threading.Lock()
    downloaded = [existing]
    t0 = time.time()

    # open file for append/write
    mode = "ab" if existing > 0 else "wb"
    with open(dest, mode) as f:
        # process chunks in order but fetch in parallel
        i = 0
        while i < len(chunks):
            batch = chunks[i:i+PARALLEL]
            results = {}
            with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
                futures = {ex.submit(fetch_chunk, url, s, e): (s, e) for s, e in batch}
                for fut in as_completed(futures):
                    start, data = fut.result()
                    results[start] = data

            # write batch in order
            for start, end in batch:
                f.write(results[start])
                downloaded[0] += len(results[start])
                elapsed = time.time() - t0 + 0.001
                speed = (downloaded[0] - existing) / elapsed / 1e6
                pct = downloaded[0] / expected_size * 100
                eta = (expected_size - downloaded[0]) / ((downloaded[0] - existing) / elapsed + 0.001)
                print(f"\r    {pct:.1f}%  {downloaded[0]/1e9:.2f}/{expected_size/1e9:.2f} GB  "
                      f"{speed:.2f} MB/s  ETA {eta/60:.0f}m  ", end="", flush=True)
            i += PARALLEL
    print()

    actual = os.path.getsize(dest)
    if actual != expected_size:
        print(f"  [WARN] size mismatch: got {actual}, expected {expected_size}")
    else:
        print(f"  [OK  ] {filename}")

def extract():
    # find part01
    part1 = os.path.join(DEST, "XJTU-SY_Bearing_Datasets.part01.rar")
    print("\n  [EXT ] Extracting (may take a few minutes)...")
    result = subprocess.run(["unrar", "x", "-y", part1, DEST + "/"], text=True)
    if result.returncode != 0:
        print(f"  [ERR ] unrar failed (exit {result.returncode})")
        sys.exit(1)
    print("  [OK  ] Extraction done")

def verify():
    print("\n  [VFY ] Checking extracted structure...")
    # walk to find bearing folders
    bearing_dirs = {}
    for root, dirs, files in os.walk(DEST):
        for d in dirs:
            if re.match(r"Bearing\d_\d", d):
                bearing_dirs[d] = os.path.join(root, d)
    if not bearing_dirs:
        print("  [FAIL] No bearing folders found after extraction!")
        return False
    total_csv = 0
    for bname in sorted(bearing_dirs)[:6]:
        csvs = [f for f in os.listdir(bearing_dirs[bname]) if f.endswith(".csv")]
        total_csv += len(csvs)
        print(f"    {bname}: {len(csvs)} CSV files")
    print(f"  Total bearing folders: {len(bearing_dirs)}, total CSVs: {total_csv}")
    return len(bearing_dirs) == 15

if __name__ == "__main__":
    print("=" * 60)
    print("  XJTU-SY Dataset Download  (MediaFire, 6-part RAR, ~4.1 GB)")
    print("  Resume-safe; re-run if interrupted.")
    print("=" * 60)

    for key, fname, size in PARTS:
        download_part(key, fname, size)

    # verify all parts complete
    missing = [fname for _, fname, size in PARTS
               if os.path.getsize(os.path.join(DEST, fname)) != size]
    if missing:
        print("\n[ERROR] Parts incomplete:", missing)
        print("Re-run this script to resume.")
        sys.exit(1)

    extract()
    ok = verify()
    if ok:
        print("\n  B4-0 DONE. Data at:", DEST)
    else:
        print("\n  [WARN] Verification incomplete — check structure manually")
