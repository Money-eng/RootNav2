import torch
from monai.metrics.surface_distance import compute_average_surface_distance

from ..base import BaseMetric

class NormalizedSurfaceDistance(BaseMetric):
    type = "gpu"

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score < old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = (prediction >= self.threshold).long()
        msk = (mask >= self.threshold).long()
        
        surface_distances = compute_average_surface_distance(
            y_pred=pred,
            y=msk,
            include_background=False,
            symmetric=True,
            distance_metric="euclidean",
        )

        avg_surface_distance = surface_distances.mean(dim=(-2, -1)) # Average over batch and channels
        return avg_surface_distance.mean().item()