import cupy as cp
import torch
from cucim.skimage import measure

from ..base import BaseMetric


def _betti0_relative_error_gpu(prediction_torch, mask_torch):
    pred = cp.from_dlpack(torch.utils.dlpack.to_dlpack((prediction_torch > 0).to(torch.uint8)))
    mask = cp.from_dlpack(torch.utils.dlpack.to_dlpack((mask_torch > 0).to(torch.uint8)))
    scores = []
    for i in range(pred.shape[0]):  # label agit image par image
        n_pred = measure.label(pred[i]).max()
        n_mask = measure.label(mask[i]).max()
        scores.append(cp.abs(n_pred - n_mask) / (n_mask + 1e-8))
    return float(cp.mean(cp.asarray(scores)).get())


class Betti0RelativeErrorGPU(BaseMetric):
    type = "gpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Relative Betti-0 Characteristic Error (RECE):
        - RECE = 0 means the prediction matches the mask perfectly.
        - The higher the RECE, the greater the difference in connected components or holes between prediction and mask.
        """
        return new_score < old_score

    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        """
        Relative Betti-0 Characteristic Error (RECE):
        RECE = |E(pred) - E(mask)| / (|E(mask)| + 1e-8)
        where E is the Betti-0 characteristic (number of connected components).
        - RECE = 0 means the prediction matches the mask perfectly.
        - The higher the RECE, the greater the difference in connected components or holes between prediction and mask.
        """
        return _betti0_relative_error_gpu(prediction, mask)
