"""案例 01（V2）：用 AnomalyDINO，資料夾怎麼放，程式就怎麼用。

資料夾結構（DATA_DIR）
    memory/             記憶庫（正常）       -> 模型只記住這些
    threshold/good/     定分界線（正常）
    threshold/bad/      定分界線（異常，可分子資料夾）
    test/good/          產線（正常）
    test/bad/           產線（異常，子資料夾名稱 = 瑕疵類別）

流程
    1. 記憶庫：用 memory/ 建立記憶庫（不需要訓練）
    2. 分界線：對 threshold/ 打分數，決定「幾分以上算瑕疵」
    3. 產線  ：對 test/ 打分數，超過分界線就判定為瑕疵，再計算成績

執行方式
    conda activate dl
    python demo_case_01_V2.py
"""

import csv
import gc
import logging
import shutil
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
# 設定區
# =============================================================================

PROJECT = Path(__file__).resolve().parent
DATA_DIR = PROJECT / "DATA" / "WM-811K" / "custom_v2"

# AnomalyDINO 的設定（每一組都會跑；不想跑的，在那一行前面加 # 註解掉）
#   encoder_name : vit_small_patch14_dinov2 / vit_base_patch14_dinov2 / vit_large_patch14_dinov2
#   masking      : True = 去掉背景雜訊
SETTINGS = {
    "small": dict(encoder_name="vit_small_patch14_dinov2", masking=False),
}

BATCH_SIZE = 16     # 一次預測幾張；顯存不夠時調小
IMAGE_SIZE = 252    # 所有影像先統一縮放成這個大小（AnomalyDINO 的輸入大小）

OUTPUT = PROJECT / "results" / "case_01_V2" / datetime.now().strftime("%Y%m%d_%H%M%S")


# =============================================================================
# 讀取資料夾
# =============================================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def find_images(folder):
    """找出資料夾（含子資料夾）裡所有的影像。"""
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def prepare(images, folder):
    """把影像統一縮放成 IMAGE_SIZE，存到工作資料夾。

    統一尺寸是因為：anomalib 把不同尺寸的圖湊成一批時會拉伸，分數會跟著「跟誰同一批」改變。
    """
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for image in images:
        resized = Image.open(image).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        resized.save(folder / f"{image.parent.name}_{image.stem}.png")
    return folder


def load_folders():
    """照資料夾結構讀取，並整理到工作資料夾。"""
    work = OUTPUT / "data"
    folders = {
        "memory": prepare(find_images(DATA_DIR / "memory"), work / "memory"),
        "threshold_good": prepare(find_images(DATA_DIR / "threshold" / "good"), work / "threshold_good"),
        "threshold_bad": prepare(find_images(DATA_DIR / "threshold" / "bad"), work / "threshold_bad"),
        "test": {"good": prepare(find_images(DATA_DIR / "test" / "good"), work / "test" / "good")},
    }
    for defect_dir in sorted(d for d in (DATA_DIR / "test" / "bad").iterdir() if d.is_dir()):
        folders["test"][defect_dir.name] = prepare(find_images(defect_dir), work / "test" / defect_dir.name)

    print(f"資料夾：{DATA_DIR}")
    print(f"  memory/            {len(find_images(folders['memory']))} 張")
    print(f"  threshold/good/    {len(find_images(folders['threshold_good']))} 張")
    print(f"  threshold/bad/     {len(find_images(folders['threshold_bad']))} 張")
    for name, folder in folders["test"].items():
        label = "test/good/" if name == "good" else f"test/bad/{name}/"
        print(f"  {label:<18} {len(find_images(folder))} 張")
    return folders


# =============================================================================
# 步驟 1：建記憶庫
# =============================================================================

def build_model(setting, folders):
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
        abnormal_dir=folders["threshold_bad"],
        normal_test_dir=folders["threshold_good"],
        train_batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        num_workers=0,  # Windows 上設 0 最穩
    )
    engine.fit(model=model, datamodule=datamodule)
    return model, engine


def score_folder(model, engine, folder):
    """對資料夾裡的影像打異常分數（分數越高越可能是瑕疵）。"""
    images = PredictDataset(folder)
    loader = DataLoader(images, batch_size=BATCH_SIZE, collate_fn=images.collate_fn, num_workers=0)
    predictions = engine.predict(model=model, dataloaders=loader, return_predictions=True)

    paths, scores = [], []
    for batch in predictions:
        paths += [str(p) for p in batch.image_path]
        scores += batch.pred_score.flatten().float().cpu().tolist()
    return paths, np.array(scores)


# =============================================================================
# 步驟 2：定分界線
# =============================================================================

def choose_threshold(good_scores, bad_scores):
    """找一個分界線，讓「正常判正常」和「瑕疵判瑕疵」的平均正確率最高。分數 >= 分界線 = 瑕疵。"""
    all_scores = np.sort(np.concatenate([good_scores, bad_scores]))
    candidates = np.concatenate([
        [all_scores[0] - 1e-6],
        (all_scores[:-1] + all_scores[1:]) / 2,
        [all_scores[-1] + 1e-6],
    ])

    def balanced_accuracy(threshold):
        return ((good_scores < threshold).mean() + (bad_scores >= threshold).mean()) / 2

    accuracies = np.array([balanced_accuracy(t) for t in candidates])
    return float(np.median(candidates[accuracies == accuracies.max()]))


# =============================================================================
# 步驟 3：產線檢測 + 評估
# =============================================================================

def run(setting_name, folders):
    model, engine = build_model(SETTINGS[setting_name], folders)

    _, good_scores = score_folder(model, engine, folders["threshold_good"])
    _, bad_scores = score_folder(model, engine, folders["threshold_bad"])
    threshold = choose_threshold(good_scores, bad_scores)

    test_scores, image_rows = {}, []
    for name, folder in folders["test"].items():
        paths, scores = score_folder(model, engine, folder)
        test_scores[name] = scores
        image_rows += [[setting_name, name, p, float(s), int(s >= threshold)] for p, s in zip(paths, scores)]

    defect_types = [name for name in test_scores if name != "good"]
    good = test_scores["good"]
    bad = np.concatenate([test_scores[t] for t in defect_types])

    result = {
        "setting": setting_name,
        "threshold": threshold,
        "auroc": roc_auc_score(np.r_[np.zeros(len(good)), np.ones(len(bad))], np.r_[good, bad]),
        "recall": float((bad >= threshold).mean()),        # 異常抓到幾成
        "false_alarm": float((good >= threshold).mean()),  # 正常被誤判幾成
    }
    for t in defect_types:
        result[f"recall[{t}]"] = float((test_scores[t] >= threshold).mean())

    del model, engine
    gc.collect()
    torch.cuda.empty_cache()
    return result, image_rows


# =============================================================================
# 主流程
# =============================================================================

def main():
    torch.set_float32_matmul_precision("high")
    folders = load_folders()

    results, image_rows = [], []
    for setting_name in SETTINGS:
        print(f"\n--- {setting_name}")
        result, rows = run(setting_name, folders)
        results.append(result)
        image_rows += rows
        print(f"    分界線={result['threshold']:.3f}  AUROC={result['auroc']:.3f}  "
              f"抓到率={result['recall']:.3f}  誤報率={result['false_alarm']:.3f}")

    with open(OUTPUT / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with open(OUTPUT / "scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["setting", "type", "image", "score", "predicted_defect"])
        writer.writerows(image_rows)
    print(f"\n結果已存到：{OUTPUT}")


if __name__ == "__main__":
    main()
