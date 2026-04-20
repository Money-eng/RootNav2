# Metrics/gpu/dice.py

import torch
from torchmetrics.segmentation import DiceScore

from .base import BaseMetric


class Dice(BaseMetric):
    type = "gpu"

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = prediction.long()
        msk = mask.long()

        Dice_metric = DiceScore(num_classes=2, average="macro", include_background=False)
        score = Dice_metric(pred, msk)
        return score.mean().item()
