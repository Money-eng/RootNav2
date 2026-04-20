import cupy as cp
import torch
from cucim.skimage import measure

from .base import BaseMetric


def _betti0_variation_index_gpu(prediction_torch, mask_torch):
    pred = cp.from_dlpack(torch.utils.dlpack.to_dlpack(prediction_torch))
    mask = cp.from_dlpack(torch.utils.dlpack.to_dlpack(mask_torch))
    scores = []
    for i in range(pred.shape[0]):  # label acts image per image
        n_pred = measure.label(pred[i], connectivity=2).max()
        n_mask = measure.label(mask[i], connectivity=2).max()
        scores.append(cp.abs(n_pred - n_mask) / (n_pred + n_mask + 1e-8))
    return float(cp.mean(cp.asarray(scores)).get())


class Betti0VariationIndexGPU(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        if abs(new_score) <= abs(old_score):
            return True
        return False

    @torch.inference_mode()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        return _betti0_variation_index_gpu(prediction, mask)
