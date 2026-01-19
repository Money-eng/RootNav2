# Metrics/gpu/pixel_accuracy.py

import torch
import torchmetrics.functional as FMF

from ..base import BaseMetric


class PixelAccuracy(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Pixel accuracy (binaire). On considère que `old_score` et `new_score`
        sont des scores de type float.
        """
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Pixel accuracy (binaire) : proportion de pixels correctement classés.
        """
        pred = prediction.float()
        msk = mask.float()

        score = FMF.accuracy(pred, msk, task="binary")
        return score.mean().item()
