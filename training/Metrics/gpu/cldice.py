# Metrics/gpu/cldice.py

import torch
from monai.losses import SoftclDiceLoss
from monai.metrics import LossMetric

from ..base import BaseMetric


class CLDice(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()
        self.cldice_loss = SoftclDiceLoss()
        self.loss_metric = LossMetric(loss_fn=self.cldice_loss)

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = prediction.float()
        msk = mask.float()

        pred_2ch = torch.cat([1 - pred, pred], dim=1)
        msk_2ch = torch.cat([1 - msk, msk], dim=1)

        loss_val = self.cldice_loss(pred_2ch, msk_2ch)
        return 1.0 - loss_val.mean().item()
