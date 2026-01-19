# Metrics/gpu/dice.py

import torch
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric

from ..base import BaseMetric


class HausdorffDistance(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Hausdorff distance. On considère que `old_score` et `new_score`
        sont des scores de type float.
        """
        return new_score < old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Compute the Hausdorff distance between the predicted segmentation and the ground truth mask.
        The Hausdorff distance is defined as the maximum distance from a point in one set to the closest point in the other set.
        """
        pred = prediction.long()
        msk = mask.long()

        metric = HausdorffDistanceMetric(
            include_background=False, distance_metric="euclidean"
        )
        metric(pred, msk)  # compute the metric on the prediction and mask tensors
        haussdorf_dist = metric.aggregate().item()  #  aggregate the results to get a single value by mean on the batch
        return haussdorf_dist
