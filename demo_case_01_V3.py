import csv
import gc
import logging
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from anomalib.data import Folder, PredictDataset
from anomalib.engine import Engine
from anomalib.models import AnomalyDINO


class AnomalyDINO:

    def __init__(self, memory_path: str, threshold_path: str) -> None:
        return
    
    pass