# Metrics/gpu/dice.py

import torch
from monai.metrics.hausdorff_distance import compute_hausdorff_distance

from .base import BaseMetric


class HausdorffDistance95(BaseMetric):
    type = "gpu"

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.pixel_size = 76 * 1e-3 

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score < old_score

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred_bin = prediction.detach().cpu().float()
        mask_bin = mask.detach().cpu().float()

        hd_tensor = compute_hausdorff_distance(
            y_pred=pred_bin,
            y=mask_bin,
            include_background=False,
            distance_metric="euclidean",
            percentile=95,
            spacing=[self.pixel_size, self.pixel_size] 
        )
        
        return float(torch.nanmean(hd_tensor).item())
