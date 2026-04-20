# Metrics/gpu/precision.py

import torch
import torchmetrics.functional as FMF

from .base import BaseMetric


class Precision(BaseMetric):
    type = "gpu"

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = prediction.float()
        msk = mask.float()

        score = FMF.precision(pred, msk, task="binary", average="macro")
        return score.mean().item()
