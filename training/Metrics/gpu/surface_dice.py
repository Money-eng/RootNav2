# Metrics/gpu/surface_difference.py

import torch
from monai.metrics.surface_dice import SurfaceDiceMetric as MonaiSurfaceDiceMetric

from ..base import BaseMetric


class Surface_dice(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Normalized Surface Dice (NSD). On considère que `old_score` et `new_score`
        sont des scores de type float.
        """
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """ 
        # Normalized Surface Dice (NSD)
        # For a reference segmentation Y and a predicted segmentation Y_hat, with
        # boundary sets ∂Y and ∂Y_hat, and a tolerance threshold τ:
        #
        # NSD(Y, Y_hat) =
        #   ( |{ p in ∂Y       : d(p, ∂Y_hat)    ≤ τ }| 
        #   + |{ q in ∂Y_hat   : d(q, ∂Y)        ≤ τ }| )
        #   ------------------------------------------------
        #   ( |∂Y| + |∂Y_hat| )
        #
        # where
        #   ∂Y       = set of boundary points of Y
        #   ∂Y_hat   = set of boundary points of Y_hat
        #   d(x, S)  = min_{s ∈ S} || x - s ||    (distance from point x to set S)
        #   |·|      = cardinality of a set
        #   τ        = tolerance threshold (in pixels or physical units)
        """

        pred = prediction.float()
        msk = mask.float()

        class_thresholds: list[float] = [0.5]  # Threshold for binarization, should be ajusted ?

        metric = MonaiSurfaceDiceMetric(include_background=False, distance_metric="euclidean",
                                        class_thresholds=class_thresholds)
        metric(pred, msk)
        surface_distance = metric.aggregate().item()
        return surface_distance
