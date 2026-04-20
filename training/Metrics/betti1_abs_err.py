# Metrics/gpu/betti0_difference.py
import cupy as cp
import torch
from cucim.skimage import measure

from .base import BaseMetric


def _betti1_abs_err(prediction_torch, mask_torch):
    pred = cp.from_dlpack(torch.utils.dlpack.to_dlpack(prediction_torch))
    mask = cp.from_dlpack(torch.utils.dlpack.to_dlpack(mask_torch))
    scores = []
    for i in range(pred.shape[0]):
        region_pred = measure.regionprops(measure.label(pred[i].astype(bool), connectivity=2))
        region_mask = measure.regionprops(measure.label(mask[i].astype(bool), connectivity=2))
        
        b0_pred = len(region_pred)  # Betti0 = number of connected components
        b0_mask = len(region_mask)
        
        euler_char_pred = sum([region.euler_number for region in region_pred]) 
        euler_char_mask = sum([region.euler_number for region in region_mask])
        
        b1_pred = b0_pred - euler_char_pred  # Betti1 = Betti0 - Euler characteristic
        b1_mask = b0_mask - euler_char_mask
        
        b1_pred = max(0, b1_pred) 
        b1_mask = max(0, b1_mask)
        scores.append(cp.abs(b1_pred - b1_mask))
    return float(cp.mean(cp.asarray(scores)).get())


class Betti1AbsErrGPU(BaseMetric):
    type = "gpu"

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    @torch.inference_mode()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        return _betti1_abs_err((prediction > self.threshold).float(), (mask > self.threshold).float())
