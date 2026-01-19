# Metrics/gpu/betti0_difference.py
import cupy as cp
import torch
from cucim.skimage import measure

from ..base import BaseMetric


def _betti0_jaccard_gpu(prediction_torch, mask_torch):
    pred = cp.from_dlpack(torch.utils.dlpack.to_dlpack((prediction_torch > 0).to(torch.uint8)))
    mask = cp.from_dlpack(torch.utils.dlpack.to_dlpack((mask_torch > 0).to(torch.uint8)))
    scores = []
    for i in range(pred.shape[0]):  # label agit image par image
        n_pred = measure.label(pred[i]).max()
        n_mask = measure.label(mask[i]).max()
        scores.append(cp.minimum(n_pred, n_mask) / (cp.maximum(n_pred, n_mask) + 1e-8))
    return float(cp.mean(cp.asarray(scores)).get())


class Betti0JaccardRatioGPU(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Jaccard Ratio of Betti-0 Characteristic (JREC) :
        - JREC = 1 means the prediction matches the mask perfectly.
        - The lower the JREC, the greater the difference in connected components between prediction and mask.
        """
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Jaccard Ratio of Betti-0 Characteristic (JREC) :
        JREC = min(|E(pred)|, |E(mask)|) / (max(|E(pred)|, |E(mask)|) + 1e-8)
        where E is the Betti-0 characteristic (number of connected components).
        - JREC = 1 means the prediction matches the mask perfectly.
        - The lower the JREC, the greater the difference in connected components between prediction and mask.
        """
        return _betti0_jaccard_gpu(prediction, mask)
