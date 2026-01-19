# Metrics/gpu/recall.py

import torch
import torchmetrics.functional as FMF

from ..base import BaseMetric


class Recall(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Recall binaire. On considère que `old_score` et `new_score`
        sont des scores de type float.
        """
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Recall binaire : TP / (TP + FN).
        """

        preds = prediction.float()
        masks = mask.float()

        score = FMF.recall(preds, masks, task="binary")
        return score.mean().item()
