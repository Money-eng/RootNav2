# Metrics/gpu/specificity.py

import torch
import torchmetrics.functional as FMF

from .base import BaseMetric


class Specificity(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = prediction.float()
        msk = mask.float()

        score = FMF.specificity(pred, msk, task="binary", average="macro")
        return score.mean().item()
