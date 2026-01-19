# Metrics/gpu/iou.py

import torch
import torchmetrics.functional.segmentation as FMF

from ..base import BaseMetric


class MeanIoU(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Intersection over Union (Jaccard). On considère que `old_score` et `new_score`
        sont des scores de type float.
        """
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Intersection over Union (Jaccard) for binary masks.
        We threshold both prediction and mask at 0.5, so that jaccard_index
        only sees exact 0/1 values.
        """
        # Ensure float
        pred = prediction.int()
        msk = mask.int()

        # Compute binary Jaccard (IoU)
        score = FMF.mean_iou(pred, msk, num_classes=2)

        return score.mean().item() if isinstance(score, torch.Tensor) else float(score)
