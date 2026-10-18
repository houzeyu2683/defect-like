"""把 WM-811K (DATA/WM-811K/LSWMD.pkl) 整理成看得懂的 CSV 表格與晶圓圖 PNG。

資料說明 (https://www.globalsino.com/ICs/page4236.html)
    811,457 片晶圓，每片是一張 2D die 陣列 (waferMap)
        0 = 不是 die (晶圓外)    1 = 正常 die    2 = 失效 die
    只有 172,950 片有標註，9 種類別：
        none, Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch

輸出 (預設在 DATA/WM-811K/export/)
    wafers.csv           全部晶圓一列一片：編號、lot、尺寸、失效 die 數、標籤、圖檔路徑
    summary.csv          各類別 × Training/Test 的數量
    images/<類別>/*.png  晶圓圖 (灰 = 正常 die、紅 = 失效 die、白 = 晶圓外)
    overview/<類別>.png  每類隨機挑幾片拼成一張總覽圖，方便快速瀏覽

執行方式
    conda activate dl
    python wm811k_export.py                       # 全部缺陷類 + none/未標註各抽 1000 張
    python wm811k_export.py --per-class 200       # 每類最多 200 張
    python wm811k_export.py --per-class 0         # 每類全部匯出 (未標註約 64 萬張，很久)
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.core.indexes
from PIL import Image

# 這個 pickle 是 Python 2 + 舊版 pandas 存的，需要舊模組路徑與 latin1 解碼
sys.modules.setdefault("pandas.indexes", pandas.core.indexes)

DEFECT_TYPES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
                "Near-full", "Random", "Scratch"]
COLORS = np.array([[255, 255, 255],   # 0 晶圓外
                   [190, 190, 190],   # 1 正常 die
                   [220, 30, 30]],    # 2 失效 die
                  dtype=np.uint8)


def unwrap(value):
    """原始標籤是 [['Training']] 或 [] 這種巢狀陣列，取出字串，空的回傳 ''。"""
    arr = np.asarray(value)
    return str(arr.ravel()[0]) if arr.size else ""


def wafer_to_image(wafer, size):
    """把 die 陣列上色並用最近鄰放大到長邊約 size 像素。"""
    rgb = COLORS[np.clip(wafer, 0, 2)]
    scale = max(1, round(size / max(wafer.shape)))
    return Image.fromarray(rgb).resize(
        (wafer.shape[1] * scale, wafer.shape[0] * scale), Image.NEAREST)


def make_overview(wafers, path, cols=8, cell=128):
    """把多片晶圓拼成 8 欄的總覽圖，檔名即類別。"""
    rows = (len(wafers) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
    for i, wafer in enumerate(wafers):
        img = wafer_to_image(wafer, cell)
        img.thumbnail((cell - 4, cell - 4), Image.NEAREST)
        x = (i % cols) * cell + (cell - img.width) // 2
        y = (i // cols) * cell + (cell - img.height) // 2
        sheet.paste(img, (x, y))
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pkl", default="DATA/WM-811K/LSWMD.pkl")
    parser.add_argument("--out", default="DATA/WM-811K/export")
    parser.add_argument("--per-class", type=int, default=None,
                        help="每類最多匯出幾張 PNG；0 = 全部。預設：缺陷類全部，none/unlabeled 各 1000")
    parser.add_argument("--size", type=int, default=256, help="PNG 長邊約幾像素")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    (out / "overview").mkdir(parents=True, exist_ok=True)

    print(f"讀取 {args.pkl} ...")
    with open(args.pkl, "rb") as f:
        df = pickle.load(f, encoding="latin1")
    print(f"共 {len(df):,} 片晶圓")

    df = df.reset_index(drop=True)
    df["wafer_id"] = np.arange(len(df))
    df["split"] = df["trianTestLabel"].map(unwrap).replace("", "unlabeled")
    df["failure_type"] = df["failureType"].map(unwrap).replace("", "unlabeled")
    df["height"] = df["waferMap"].map(lambda w: w.shape[0])
    df["width"] = df["waferMap"].map(lambda w: w.shape[1])
    df["n_dies"] = df["waferMap"].map(lambda w: int((w > 0).sum()))
    df["n_fail_dies"] = df["waferMap"].map(lambda w: int((w == 2).sum()))
    df["fail_ratio"] = (df["n_fail_dies"] / df["n_dies"].where(df["n_dies"] > 0)).round(4)
    df["waferIndex"] = df["waferIndex"].astype(int)
    df["image_path"] = ""

    # 決定哪些晶圓要輸出 PNG
    rng = np.random.default_rng(args.seed)
    for cls, idx in df.groupby("failure_type").groups.items():
        idx = np.asarray(idx)
        if args.per_class is None:
            limit = 1000 if cls in ("none", "unlabeled") else 0
        else:
            limit = args.per_class
        if limit and len(idx) > limit:
            idx = np.sort(rng.choice(idx, limit, replace=False))
        folder = out / "images" / cls
        folder.mkdir(parents=True, exist_ok=True)
        for n, i in enumerate(idx, 1):
            row = df.loc[i]
            name = f"{row.wafer_id:06d}_{row.lotName}_w{row.waferIndex:02d}.png"
            wafer_to_image(row.waferMap, args.size).save(folder / name)
            df.at[i, "image_path"] = (folder / name).relative_to(out).as_posix()
            if n % 5000 == 0:
                print(f"  {cls}: {n:,}/{len(idx):,}")
        print(f"{cls:>10}: 輸出 {len(idx):,} 張 PNG")

        pick = rng.choice(np.asarray(df.groupby("failure_type").groups[cls]),
                          min(64, len(idx)), replace=False)
        make_overview([df.at[i, "waferMap"] for i in pick], out / "overview" / f"{cls}.png")

    columns = ["wafer_id", "lotName", "waferIndex", "split", "failure_type", "dieSize",
               "height", "width", "n_dies", "n_fail_dies", "fail_ratio", "image_path"]
    df[columns].rename(columns={"lotName": "lot_name", "waferIndex": "wafer_index",
                                "dieSize": "die_size"}) \
        .to_csv(out / "wafers.csv", index=False, encoding="utf-8-sig")

    summary = pd.crosstab(df["failure_type"], df["split"], margins=True, margins_name="total")
    summary.to_csv(out / "summary.csv", encoding="utf-8-sig")
    print(summary)
    print(f"\n完成，輸出在 {out.resolve()}")


if __name__ == "__main__":
    main()
