# Metrics/gpu/f1_score.py

import torch
import torchmetrics.functional as FMF

from ..base import BaseMetric


class F1Score(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        F1 Score binaire. On considère que `old_score` et `new_score`
        sont des scores de type float.
        """
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        F1 Score binaire. Equivalent au Dice dans la plupart des cas,
        mais on utilise torchmetrics.functional.f1_score().
        """
        pred = prediction.float()
        msk = mask.float()

        # FMF.f1_score retourne un tenseur, on fait mean().item()
        score = FMF.f1_score(pred, msk, average="micro", task="binary")
        return score.mean().item()
