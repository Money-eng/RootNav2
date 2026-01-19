import cupy as cp
import torch
from cucim.skimage import measure

from ..base import BaseMetric


def _betti0_variation_index_gpu(prediction_torch, mask_torch):
    pred = cp.from_dlpack(torch.utils.dlpack.to_dlpack((prediction_torch > 0).to(torch.uint8)))
    mask = cp.from_dlpack(torch.utils.dlpack.to_dlpack((mask_torch > 0).to(torch.uint8)))
    scores = []
    for i in range(pred.shape[0]):  # label agit image par image
        n_pred = measure.label(pred[i]).max()
        n_mask = measure.label(mask[i]).max()
        scores.append(cp.abs(n_pred - n_mask) / (n_pred + n_mask + 1e-8))
    return float(cp.mean(cp.asarray(scores)).get())


class Betti0VariationIndexGPU(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Variation of Betti-0 Characteristic (VEC):
        VEC = (E(pred) - E(mask)) / (E(pred) + E
        - VEC = 0 means the prediction matches the mask perfectly.
        - VEC > 0 means the prediction has more connected components or holes than the mask.
        - VEC < 0 means the prediction has fewer connected components or holes than the mask.
        """
        if abs(new_score) <= abs(old_score):
            return True
        return False

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Variation of Betti-0 Characteristic (VEC):
        VEC = (E(pred) - E(mask)) / (E(pred) + E(mask))
        where E is the Betti-0 characteristic (number of connected components).
        
        - VEC = 0 means the prediction matches the mask perfectly.
        - VEC > 0 means the prediction has more connected components or holes than the mask.
        - VEC < 0 means the prediction has fewer connected components or holes than the mask.
        """
        return _betti0_variation_index_gpu(prediction, mask)
