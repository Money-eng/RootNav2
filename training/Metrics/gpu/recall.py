# Metrics/gpu/recall.py

import torch
import torchmetrics.functional as FMF

from ..base import BaseMetric


class Recall(BaseMetric):
    type = "gpu"

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        preds = (prediction > self.threshold).long()
        masks = (mask > self.threshold).long()

        score = FMF.recall(preds, masks, task="binary")
        return score.mean().item()
