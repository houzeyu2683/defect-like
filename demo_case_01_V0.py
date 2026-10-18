"""案例 01：只用少量過去的影像，偵測產線上「從沒看過」的瑕疵。

情境（見 cases/01.md）
    資料庫裡只有少量正常影像、少量過去出現過的瑕疵影像。
    產線上出現了一種新的、從沒看過的瑕疵，能不能把它抓出來？

資料夾結構（DATA_DIR）
    good/          過去的正常圖 -> 前 SHOTS 張進記憶庫，剩下的用來定分界線
    bad/           過去的異常圖 -> 全部用來定分界線（有沒有分子資料夾都可以）
    test/good/     產線上的正常圖
    test/bad/      產線上的異常圖（子資料夾名稱 = 瑕疵類別，用來分別統計抓到率）

    good/ 和 bad/ 是「過去的資料」，test/ 是「要檢查的新資料」。

流程
    1. 記憶庫：模型只看 good/ 裡的 SHOTS 張正常圖，記住「正常長這樣」（不需要訓練）
    2. 分界線：用 good/ 剩下的正常圖和 bad/ 的異常圖打分數，決定「幾分以上算瑕疵」
    3. 產線  ：對 test/ 的每張圖打分數，超過分界線就判定為瑕疵，再計算成績

執行方式
    conda activate dl
    python demo_case_01.py

要改實驗內容，只要修改下面「設定區」的數值。
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
from sklearn.metrics import roc_auc_score

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import AnomalyDINO, Patchcore, Rad, SuperADD


# =============================================================================
# 設定區：要改實驗，改這裡就好
# =============================================================================

PROJECT = Path(__file__).resolve().parent

# ---- 資料 ----
DATA_DIR = PROJECT / "DATA" / "WM-811K" / "custom"   # 照上面說明的結構放好照片
SHOTS = [1, 3, 5]   # good/ 裡幾張放進記憶庫（每個數字跑一次）；剩下的用來定分界線
SEED = 0            # 隨機抽圖的種子，換數字 = 換一批圖

# ---- 要比較的模型（不想跑的，在那一段前面加 # 註解掉）----
# 每個模型的參數都可以直接改，說明見右邊註解。
MODELS = {
    "patchcore": dict(
        backbone="wide_resnet50_2",      # 抽特徵的網路，例如 resnet18（較快）
        coreset_sampling_ratio=0.1,      # 記憶庫保留幾成特徵；正常圖很少時可試 1.0
        num_neighbors=9,                 # 打分數時參考幾個最近的特徵
    ),
    "anomalydino": dict(
        encoder_name="vit_small_patch14_dinov2",  # 可換 vit_base_patch14_dinov2 / vit_large_patch14_dinov2
        masking=False,                            # True = 去掉背景雜訊
        num_neighbours=1,
    ),
    "rad": dict(
        backbone="vit_base_patch16_dinov3",
        use_positional_bank=True,        # True = 只跟「相同位置」的正常區塊比；產品位置不固定時可試 False
    ),
    "superadd": dict(
        backbone="vit_huge_plus_patch16_dinov3",  # 很大（約 3.4 GB）；可換 vit_base_patch16_dinov3 加快
        gaussian_blur_sigma=4.0,                  # 異常熱力圖的平滑程度
    ),
}

# ---- 輸出 ----
RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")    # 每次執行存到不同資料夾，不會互相覆蓋
OUTPUT = PROJECT / "results" / "case_01" / RUN_NAME


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
# 步驟 1：把過去的資料分成「記憶庫」和「定分界線用」
# =============================================================================

def copy_images(images, folder):
    """把指定的影像複製到一個新資料夾（檔名前加上來源資料夾，避免重名）。"""
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for image in images:
        shutil.copy(image, folder / f"{image.parent.name}_{image.name}")
    return folder


def split_past_data(dataset, shots):
    normal_images = list(dataset["good"])
    random.Random(SEED).shuffle(normal_images)

    work_dir = OUTPUT / "data" / f"k{shots}"
    return {
        "memory": copy_images(normal_images[:shots], work_dir / "memory"),              # 記憶庫
        "threshold_normal": copy_images(normal_images[shots:], work_dir / "threshold_normal"),  # 定分界線：正常
        "threshold_defect": copy_images(dataset["bad"], work_dir / "threshold_defect"),  # 定分界線：異常
    }


# =============================================================================
# 步驟 2：建立模型的記憶庫（只看正常圖，不需要訓練）
# =============================================================================

MODEL_CLASSES = {"patchcore": Patchcore, "anomalydino": AnomalyDINO, "rad": Rad, "superadd": SuperADD}


def build_model(model_name, folders):
    """建立模型，並用記憶庫裡的正常圖建好記憶庫。"""
    # post_processor 等關掉：讓模型輸出原始分數，分界線由我們自己決定
    model = MODEL_CLASSES[model_name](
        **MODELS[model_name],
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
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
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,  # Windows 上設 0 最穩
    )
    engine.fit(model=model, datamodule=datamodule)
    return model, engine


def score_images(model, engine, images):
    """對一組影像打異常分數（分數越高越可能是瑕疵）。"""
    scores = []
    for image in images:
        predictions = engine.predict(model=model, data_path=image, return_predictions=True)
        scores.append(float(predictions[0].pred_score.flatten()[0]))
    return np.array(scores)


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

def run_experiment(dataset, shots, model_name):
    """跑一組實驗：建記憶庫 -> 定分界線 -> 產線檢測 -> 計算成績。"""
    folders = split_past_data(dataset, shots)

    # 建記憶庫
    start = time.time()
    model, engine = build_model(model_name, folders)
    build_seconds = time.time() - start

    # 定分界線
    normal_scores = score_images(model, engine, find_images(folders["threshold_normal"]))
    defect_scores = score_images(model, engine, find_images(folders["threshold_defect"]))
    threshold, past_accuracy = choose_threshold(normal_scores, defect_scores)

    # 產線檢測
    start = time.time()
    test_sets = {"good": dataset["test_good"]} | dataset["test_bad"]
    test_scores = {name: score_images(model, engine, images) for name, images in test_sets.items()}
    image_count = sum(len(images) for images in test_sets.values())
    ms_per_image = (time.time() - start) / max(image_count, 1) * 1000

    image_rows = [
        [model_name, shots, name, str(image), float(score), int(score >= threshold)]
        for name, images in test_sets.items()
        for image, score in zip(images, test_scores[name])
    ]

    # 計算成績
    good_scores = test_scores["good"]
    defect_types = list(dataset["test_bad"])
    bad_scores = np.concatenate([test_scores[t] for t in defect_types])
    labels = np.r_[np.zeros(len(good_scores)), np.ones(len(bad_scores))]

    result = {
        "model": model_name,
        "shots": shots,
        "auroc": roc_auc_score(labels, np.r_[good_scores, bad_scores]),  # 能不能分開（不看分界線）
        "recall": float((bad_scores >= threshold).mean()),               # 異常圖抓到幾成
        "false_alarm": float((good_scores >= threshold).mean()),         # 正常圖被誤判幾成
        "past_accuracy": past_accuracy,                                  # 分界線在過去資料上的正確率
        "threshold": threshold,
        "build_seconds": build_seconds,
        "ms_per_image": ms_per_image,
    }
    for t in defect_types:
        result[f"recall[{t}]"] = float((test_scores[t] >= threshold).mean())

    # 釋放顯存，避免下一個模型不夠用
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
    print("\n========== 總結 ==========")
    print(f"{'模型':<12} {'SHOTS':>5}  {'AUROC':>6}  {'抓到率':>6}  {'誤報率':>6}  {'ms/張':>6}")
    for r in results:
        print(
            f"{r['model']:<14} {r['shots']:>5}  {r['auroc']:>6.3f}  {r['recall']:>9.3f}  "
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
        writer.writerow(["model", "shots", "type", "image", "score", "predicted_defect"])
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
    results, image_rows = [], []
    for shots in SHOTS:
        if shots >= len(dataset["good"]):
            print(f"\nSHOTS={shots} 跳過：good/ 只有 {len(dataset['good'])} 張，至少要留 1 張定分界線。")
            continue
        print(f"\n=== 記憶庫 {shots} 張，定分界線用 {len(dataset['good']) - shots} 張正常 + {len(dataset['bad'])} 張異常")

        for model_name in MODELS:
            print(f"--- {model_name} ...", flush=True)
            try:
                result, rows = run_experiment(dataset, shots, model_name)
            except Exception as error:  # 某個模型失敗（例如顯存不足）時，繼續跑下一個
                print(f"    失敗：{type(error).__name__}: {error}")
                gc.collect()
                torch.cuda.empty_cache()
                continue

            results.append(result)
            image_rows += rows
            print_result(result, list(dataset["test_bad"]))

    if results:
        print_summary(results)
        save_results(results, image_rows)


if __name__ == "__main__":
    main()
