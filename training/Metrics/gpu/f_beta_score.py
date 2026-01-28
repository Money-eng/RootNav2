# Metrics/gpu/f1_score.py

import torch
import torchmetrics.functional as FMF

from ..base import BaseMetric


class FBetaScore(BaseMetric):
    type = "gpu"

    def __init__(self, beta: float = 2.0, threshold: float = 0.5):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = (prediction > self.threshold).long()
        msk = (mask > self.threshold).long()

        score = FMF.fbeta_score(pred, msk, beta=self.beta, average="micro", task="binary")
        return score.mean().item()