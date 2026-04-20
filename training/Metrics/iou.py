# Metrics/gpu/iou.py

import torch
from monai.metrics import compute_iou

from .base import BaseMetric


class IoU(BaseMetric):
    type = "gpu"

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    @torch.inference_mode()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = prediction.float()
        msk = mask.float()

        iou_tensor = compute_iou(y_pred=pred, y=msk, ignore_empty=True)
        
        return float(torch.nanmean(iou_tensor).item()) 
