import cupy as cp
import torch
from cucim.skimage import measure

from ..base import BaseMetric


def _betti1_relative_error_gpu(prediction_torch, mask_torch):
    pred = cp.from_dlpack(torch.utils.dlpack.to_dlpack((prediction_torch > 0).to(torch.uint8)))
    mask = cp.from_dlpack(torch.utils.dlpack.to_dlpack((mask_torch > 0).to(torch.uint8)))
    scores = []
    for i in range(pred.shape[0]):
        n_pred = measure.label(~pred[i].astype(bool)).max()  # inverted image connected components
        n_mask = measure.label(~mask[i].astype(bool)).max()
        b1_pred = max(0, n_pred - 1)  # Betti1 = number of holes = cc_inverted - 1
        b1_mask = max(0, n_mask - 1)
        scores.append(cp.abs(b1_pred - b1_mask) / (b1_mask + 1e-8))
    return float(cp.mean(cp.asarray(scores)).get())


class Betti1RelativeErrorGPU(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        return _betti1_relative_error_gpu(prediction, mask)
