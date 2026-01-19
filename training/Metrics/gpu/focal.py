# Metrics/gpu/cldice.py

import torch
from monai.losses.focal_loss import FocalLoss as FC_loss
from monai.metrics import LossMetric

from ..base import BaseMetric


class FocalLoss(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = prediction.float()
        msk = mask.float()

        focal_loss_fn = FC_loss()
        loss_metric = LossMetric(loss_fn=focal_loss_fn)

        loss_metric(pred, msk)
        score = loss_metric.aggregate(reduction="mean")
        return score.item()
