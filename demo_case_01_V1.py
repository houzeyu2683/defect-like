"""案例 01（V1）：專注在 AnomalyDINO，比較不同設定的效果。

和 V0 的差別
    - 只用 AnomalyDINO，在設定區列出多組參數，一次比較
    - 整批預測（V0 是一張一張預測），速度快很多

資料夾結構（DATA_DIR），和 V0 相同
    good/          過去的正常圖 -> 前 SHOTS 張進記憶庫，剩下的用來定分界線
    bad/           過去的異常圖 -> 全部用來定分界線（有沒有分子資料夾都可以）
    test/good/     產線上的正常圖
    test/bad/      產線上的異常圖（子資料夾名稱 = 瑕疵類別，用來分別統計抓到率）

流程（每個 SHOTS × 每組設定 跑一次）
    1. 記憶庫：模型只看 good/ 裡的 SHOTS 張正常圖，記住「正常長這樣」（不需要訓練）
    2. 分界線：用 good/ 剩下的正常圖和 bad/ 的異常圖打分數，決定「幾分以上算瑕疵」
    3. 產線  ：對 test/ 的每張圖打分數，超過分界線就判定為瑕疵，再計算成績

執行方式
    conda activate dl
    python demo_case_01_V1.py
"""

import csv
import gc
import logging
import random
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
for name in ("lightning", "lightning.pytorch", "anomalib"):
    logging.getLogger(name).setLevel(logging.ERROR)

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from anomalib.data import Folder, PredictDataset
from anomalib.engine import Engine
from anomalib.models import AnomalyDINO


# =============================================================================
# 設定區：要改實驗，改這裡就好
# =============================================================================

PROJECT = Path(__file__).resolve().parent

# ---- 資料 ----
DATA_DIR = PROJECT / "DATA" / "WM-811K" / "custom"   # 照上面說明的結構放好照片
SHOTS = [1, 3, 5]   # good/ 裡幾張放進記憶庫（每個數字跑一次）；剩下的用來定分界線
SEED = 0            # 隨機抽圖的種子，換數字 = 換一批圖

# ---- AnomalyDINO 的設定（每一組都會跑；不想跑的，在那一行前面加 # 註解掉）----
#   encoder_name   : 抽特徵的網路大小
#                    vit_small_patch14_dinov2（小，約 0.09 GB）
#                    vit_base_patch14_dinov2 （中，約 0.35 GB）
#                    vit_large_patch14_dinov2（大，約 1.2 GB）
#   masking        : True = 去掉背景雜訊（例如晶圓圓形以外的白色區域）
#   num_neighbours : 打分數時參考幾個最近的正常區塊
SMALL = "vit_small_patch14_dinov2"
BASE = "vit_base_patch14_dinov2"
LARGE = "vit_large_patch14_dinov2"

SETTINGS = {
    "small":       dict(encoder_name=SMALL, masking=False, num_neighbours=1),   # 預設值（= V0）
    "small+mask":  dict(encoder_name=SMALL, masking=True,  num_neighbours=1),
    "base":        dict(encoder_name=BASE,  masking=False, num_neighbours=1),
    "base+mask":   dict(encoder_name=BASE,  masking=True,  num_neighbours=1),
    "large":       dict(encoder_name=LARGE, masking=False, num_neighbours=1),
    "large+mask":  dict(encoder_name=LARGE, masking=True,  num_neighbours=1),
}

BATCH_SIZE = 16     # 一次預測幾張；顯存不夠時調小
IMAGE_SIZE = 252    # 所有影像先統一縮放成這個大小（AnomalyDINO 的輸入大小就是 252）
#                     用「最近鄰」縮放，晶圓圖的方格不會被糊掉

# ---- 輸出 ----
RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")    # 每次執行存到不同資料夾，不會互相覆蓋
OUTPUT = PROJECT / "results" / "case_01_V1" / RUN_NAME


# =============================================================================
# 步驟 0：讀取資料夾
# =============================================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_images(folder):
    """找出資料夾（含子資料夾）裡所有的影像。"""
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def find_test_defects(folder):
    """test/bad/ 裡的異常圖，依子資料夾名稱分成各個瑕疵類別。"""
    by_type = {}
    for image in find_images(folder):
        relative = image.relative_to(folder)
        defect_type = relative.parts[0] if len(relative.parts) > 1 else "bad"
        by_type.setdefault(defect_type, []).append(image)
    return by_type


def load_dataset():
    dataset = {
        "good": find_images(DATA_DIR / "good"),
        "bad": find_images(DATA_DIR / "bad"),
        "test_good": find_images(DATA_DIR / "test" / "good"),
        "test_bad": find_test_defects(DATA_DIR / "test" / "bad"),
    }
    print(f"資料夾：{DATA_DIR}")
    print(f"  過去的正常圖  good/       {len(dataset['good'])} 張")
    print(f"  過去的異常圖  bad/        {len(dataset['bad'])} 張")
    print(f"  產線正常圖    test/good/  {len(dataset['test_good'])} 張")
    for defect_type, images in dataset["test_bad"].items():
        print(f"  產線異常圖    test/bad/{defect_type:<10} {len(images)} 張")
    return dataset


# =============================================================================
# 步驟 1：把資料整理成幾個資料夾（記憶庫 / 定分界線用 / 產線）
# =============================================================================

def copy_images(images, folder):
    """把影像統一縮放成 IMAGE_SIZE，存到一個新資料夾（檔名前加上來源資料夾，避免重名）。

    為什麼要先統一尺寸：anomalib 把不同尺寸的圖湊成一批時，會把整批拉伸成最大那張的尺寸，
    導致同一張圖的分數會隨「跟誰分在同一批」而改變。先統一尺寸就不會有這個問題。
    """
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for image in images:
        resized = Image.open(image).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        resized.save(folder / f"{image.parent.name}_{image.stem}.png")
    return folder


def prepare_folders(dataset, shots):
    """依照 SHOTS 切分過去的資料；產線資料每個類別一個資料夾，方便整批預測。"""
    normal_images = list(dataset["good"])
    random.Random(SEED).shuffle(normal_images)

    work_dir = OUTPUT / "data" / f"k{shots}"
    folders = {
        "memory": copy_images(normal_images[:shots], work_dir / "memory"),                # 記憶庫
        "threshold_normal": copy_images(normal_images[shots:], work_dir / "threshold_normal"),  # 定分界線：正常
        "threshold_defect": copy_images(dataset["bad"], work_dir / "threshold_defect"),  # 定分界線：異常
        "test": {"good": copy_images(dataset["test_good"], work_dir / "test" / "good")},  # 產線
    }
    for defect_type, images in dataset["test_bad"].items():
        folders["test"][defect_type] = copy_images(images, work_dir / "test" / defect_type)
    return folders


# =============================================================================
# 步驟 2：建立 AnomalyDINO 的記憶庫（只看正常圖，不需要訓練）
# =============================================================================

def build_model(setting, folders):
    """建立模型，並用記憶庫裡的正常圖建好記憶庫。"""
    # post_processor 等關掉：讓模型輸出原始分數，分界線由我們自己決定
    model = AnomalyDINO(**setting, post_processor=False, evaluator=False, visualizer=False)
    engine = Engine(
        default_root_dir=OUTPUT / "lightning",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    datamodule = Folder(
        name="custom",
        normal_dir=folders["memory"],
        abnormal_dir=folders["threshold_defect"],
        normal_test_dir=folders["threshold_normal"],
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        num_workers=0,  # Windows 上設 0 最穩
    )
    engine.fit(model=model, datamodule=datamodule)
    return model, engine


def score_folder(model, engine, folder):
    """對一個資料夾裡的影像整批打異常分數（分數越高越可能是瑕疵）。"""
    images = PredictDataset(folder)
    loader = DataLoader(images, batch_size=BATCH_SIZE, collate_fn=images.collate_fn, num_workers=0)
    predictions = engine.predict(model=model, dataloaders=loader, return_predictions=True)

    paths, scores = [], []
    for batch in predictions:
        paths += [str(p) for p in batch.image_path]
        scores += batch.pred_score.flatten().float().cpu().tolist()
    return paths, np.array(scores)


# =============================================================================
# 步驟 3：用過去的資料（正常圖 + 異常圖）決定分界線
# =============================================================================

def choose_threshold(normal_scores, defect_scores):
    """找一個分界線，讓「正常判正常」和「瑕疵判瑕疵」的平均正確率最高。

    分數 >= 分界線 就判定為瑕疵。
    """
    all_scores = np.sort(np.concatenate([normal_scores, defect_scores]))
    # 候選分界線：相鄰兩個分數的中間點，再加上最低和最高兩端
    candidates = np.concatenate([
        [all_scores[0] - 1e-6],
        (all_scores[:-1] + all_scores[1:]) / 2,
        [all_scores[-1] + 1e-6],
    ])

    def balanced_accuracy(threshold):
        normal_correct = (normal_scores < threshold).mean()
        defect_correct = (defect_scores >= threshold).mean()
        return (normal_correct + defect_correct) / 2

    accuracies = np.array([balanced_accuracy(t) for t in candidates])
    best_candidates = candidates[accuracies == accuracies.max()]
    return float(np.median(best_candidates)), float(accuracies.max())


# =============================================================================
# 步驟 4：產線檢測 + 評估
# =============================================================================

def run_experiment(folders, shots, setting_name):
    """跑一組實驗：建記憶庫 -> 定分界線 -> 產線檢測 -> 計算成績。"""
    # 建記憶庫
    start = time.time()
    model, engine = build_model(SETTINGS[setting_name], folders)
    build_seconds = time.time() - start

    # 定分界線
    _, normal_scores = score_folder(model, engine, folders["threshold_normal"])
    _, defect_scores = score_folder(model, engine, folders["threshold_defect"])
    threshold, past_accuracy = choose_threshold(normal_scores, defect_scores)

    # 產線檢測
    start = time.time()
    test_scores, image_rows = {}, []
    for name, folder in folders["test"].items():
        paths, scores = score_folder(model, engine, folder)
        test_scores[name] = scores
        for path, score in zip(paths, scores):
            image_rows.append([setting_name, shots, name, path, float(score), int(score >= threshold)])
    image_count = sum(len(s) for s in test_scores.values())
    ms_per_image = (time.time() - start) / max(image_count, 1) * 1000

    # 計算成績
    defect_types = [name for name in folders["test"] if name != "good"]
    good_scores = test_scores["good"]
    bad_scores = np.concatenate([test_scores[t] for t in defect_types])
    labels = np.r_[np.zeros(len(good_scores)), np.ones(len(bad_scores))]

    result = {
        "setting": setting_name,
        "shots": shots,
        "auroc": roc_auc_score(labels, np.r_[good_scores, bad_scores]),  # 能不能分開（不看分界線）
        "recall": float((bad_scores >= threshold).mean()),               # 異常圖抓到幾成
        "false_alarm": float((good_scores >= threshold).mean()),         # 正常圖被誤判幾成
        "past_accuracy": past_accuracy,                                  # 分界線在過去資料上的正確率
        "threshold": threshold,
        "build_seconds": build_seconds,
        "ms_per_image": ms_per_image,
    } | SETTINGS[setting_name]
    for t in defect_types:
        result[f"recall[{t}]"] = float((test_scores[t] >= threshold).mean())

    # 釋放顯存，避免下一組不夠用
    del model, engine
    gc.collect()
    torch.cuda.empty_cache()
    return result, image_rows


# =============================================================================
# 輸出結果
# =============================================================================

def print_result(result, defect_types):
    print(
        f"    AUROC={result['auroc']:.3f}  抓到率={result['recall']:.3f}  誤報率={result['false_alarm']:.3f}  "
        f"建庫={result['build_seconds']:.1f}s  {result['ms_per_image']:.0f} ms/張"
    )
    per_type = "  ".join(f"{t}={result[f'recall[{t}]']:.2f}" for t in defect_types)
    print(f"    各類抓到率：{per_type}")


def print_summary(results):
    print("\n========== 總結（依 AUROC 由高到低）==========")
    print(f"{'設定':<12} {'SHOTS':>5}  {'AUROC':>6}  {'抓到率':>6}  {'誤報率':>6}  {'ms/張':>6}")
    for r in sorted(results, key=lambda r: -r["auroc"]):
        print(
            f"{r['setting']:<14} {r['shots']:>5}  {r['auroc']:>6.3f}  {r['recall']:>9.3f}  "
            f"{r['false_alarm']:>9.3f}  {r['ms_per_image']:>7.0f}"
        )


def save_results(results, image_rows):
    columns = list(dict.fromkeys(key for r in results for key in r))
    with open(OUTPUT / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    with open(OUTPUT / "scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "shots", "type", "image", "score", "predicted_defect"])
        writer.writerows(image_rows)

    print(f"\n結果已存到：{OUTPUT}")


# =============================================================================
# 主流程
# =============================================================================

def main():
    torch.set_float32_matmul_precision("high")
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"使用裝置：{device}")

    dataset = load_dataset()
    if not dataset["good"] or not dataset["bad"] or not dataset["test_good"] or not dataset["test_bad"]:
        print("資料夾裡缺少 good/、bad/、test/good/ 或 test/bad/ 的影像，請先放好再執行。")
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    defect_types = list(dataset["test_bad"])
    results, image_rows = [], []
    for shots in SHOTS:
        if shots >= len(dataset["good"]):
            print(f"\nSHOTS={shots} 跳過：good/ 只有 {len(dataset['good'])} 張，至少要留 1 張定分界線。")
            continue
        print(f"\n=== 記憶庫 {shots} 張，定分界線用 {len(dataset['good']) - shots} 張正常 + {len(dataset['bad'])} 張異常")
        folders = prepare_folders(dataset, shots)

        for setting_name in SETTINGS:
            print(f"--- {setting_name} ...", flush=True)
            try:
                result, rows = run_experiment(folders, shots, setting_name)
            except Exception as error:  # 某組設定失敗（例如顯存不足）時，繼續跑下一組
                print(f"    失敗：{type(error).__name__}: {error}")
                gc.collect()
                torch.cuda.empty_cache()
                continue

            results.append(result)
            image_rows += rows
            print_result(result, defect_types)

    if results:
        print_summary(results)
        save_results(results, image_rows)


if __name__ == "__main__":
    main()
