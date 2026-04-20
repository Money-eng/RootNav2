# Metrics/gpu/f1_score.py

import torch
import torchmetrics.functional as FMF

from .base import BaseMetric


class F1Score(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    @torch.inference_mode()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = prediction.float()
        msk = mask.float()

        score = FMF.f1_score(pred, msk, average="macro", task="binary")
        return score.mean().item()
