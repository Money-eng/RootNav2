# Metrics/gpu/mean_iou.py

import torch
from torchmetrics.functional import jaccard_index
from .base import BaseMetric


class MeanIoU(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = prediction.long()
        msk = mask.long()

        score = jaccard_index(
            pred, 
            msk, 
            task="multiclass", 
            num_classes=2, 
            average="macro"
        )

        return float(score.item())