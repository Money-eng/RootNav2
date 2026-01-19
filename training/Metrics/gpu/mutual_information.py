# Metrics/gpu/ari_index.py

import torch
from torchmetrics.clustering import NormalizedMutualInfoScore

from ..base import BaseMetric


class NormalizedMutualInformation(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Normalized Mutual Information (NMI). On considère que `old_score` et `new_score`
        sont des scores de type float.
        """
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred_np = prediction.long().flatten()
        mask_np = mask.long().flatten()

        NMI = NormalizedMutualInfoScore("geometric")
        return float(NMI(pred_np, mask_np).item())
