# Metrics/gpu/surface_difference.py

import torch
from monai.metrics.surface_distance import SurfaceDistanceMetric as MonaiSurfaceDistanceMetric

from ..base import BaseMetric


class Surface_distance(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Surface distance. On considère que `old_score` et `new_score`
        sont des scores de type float.
        """
        return new_score < old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """ 
        Compute the surface distance between the predicted segmentation and the ground truth mask.
            SD(pred, mask) = 1/δpred SUM_(border pred : p)( min_(border mask : q) || p - q ||_2 )
        In other words, it computes the sum of the minimum Euclidean distances from each point on the border of the predicted segmentation to the nearest point on the border of the ground truth mask, averaged over all batch elements.
        
        # Pour un masque binaire X ⊂ ℤ^d, on définit sa frontière :
        #   ∂X = { p ∈ X : ∃ q ∉ X tel que ‖p - q‖₁ = 1 }
        #
        # Pour deux masques M (prédiction) et G (vérité terrain) :
        #
        #   d(p, ∂G) = min_{q ∈ ∂G} ‖p - q‖₂
        #     où ‖·‖₂ est la norme euclidienne
        #
        # ASD asymétrique :
        #   ASD(M, G)
        #     = (1 / |∂M|) · ∑_{p ∈ ∂M} d(p, ∂G)
        #
        # ASD symétrique :
        #   ASD_sym(M, G)
        #     = ½ [ ASD(M, G) + ASD(G, M) ]
        #
        # Pour un batch de taille B et C classes (maskes one-hot binarisés) :
        #   ASD[b, c] = ASD( M_c^(b), G_c^(b) )
        #     avec éventuellement exclusion de la classe de fond c = 0
        """
        pred = prediction.float()
        msk = mask.float()

        metric = MonaiSurfaceDistanceMetric(include_background=False, distance_metric="euclidean")  # define the metric
        metric(pred, msk)  # compute the metric on the prediction and mask tensors
        surface_distance = metric.aggregate().item()  #  aggregate the results to get a single value by mean on the batch
        return surface_distance
