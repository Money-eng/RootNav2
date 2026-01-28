# Metrics/gpu/dice.py

import torch
import torchmetrics.functional.segmentation as FMS

from ..base import BaseMetric


class Dice(BaseMetric):
    type = "gpu"

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = (prediction > self.threshold).long()
        msk = (mask > self.threshold).long()

        score = FMS.dice_score(pred, msk, num_classes=2, average="micro")
        return score.mean().item()
