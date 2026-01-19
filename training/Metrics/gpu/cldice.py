# Metrics/gpu/cldice.py

import torch
from monai.losses import SoftclDiceLoss
from monai.metrics import LossMetric

from ..base import BaseMetric


class CLDice(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = prediction.float()
        msk = mask.float()

        cldice_loss = SoftclDiceLoss()
        loss_metric = LossMetric(loss_fn=cldice_loss)

        loss_metric(pred, msk)
        score = loss_metric.aggregate(reduction="mean")
        return score.item()
