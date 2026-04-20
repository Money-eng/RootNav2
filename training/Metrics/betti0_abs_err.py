# Metrics/gpu/betti0_difference.py
import cupy as cp
import torch
from cucim.skimage import measure

from .base import BaseMetric


def _betti0_abs_err(prediction_torch, mask_torch):
    pred = cp.from_dlpack(torch.utils.dlpack.to_dlpack(prediction_torch))
    mask = cp.from_dlpack(torch.utils.dlpack.to_dlpack(mask_torch))
    scores = []
    for i in range(pred.shape[0]):  # label agit image par image
        n_pred = measure.label(pred[i], connectivity=2).max()
        n_mask = measure.label(mask[i], connectivity=2).max()
        scores.append(cp.abs(n_pred - n_mask))
    return float(cp.mean(cp.asarray(scores)).get())


class Betti0AbsErrGPU(BaseMetric):
    type = "gpu"

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        return _betti0_abs_err((prediction > self.threshold).float(), (mask > self.threshold).float())
