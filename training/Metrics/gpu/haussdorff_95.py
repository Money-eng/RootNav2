# Metrics/gpu/dice.py

import torch
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric

from ..base import BaseMetric


class HausdorffDistance95(BaseMetric):
    type = "gpu"

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score < old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = (prediction > self.threshold).detach().cpu().long()
        msk = (mask > self.threshold).detach().cpu().long()

        metric = HausdorffDistanceMetric(
            include_background=True, distance_metric="euclidean", percentile=95
        )
        metric(pred, msk)  # compute the metric on the prediction and mask tensors
        haussdorf_dist = metric.aggregate().item()  #  aggregate the results to get a single value by mean on the batch
        return haussdorf_dist
