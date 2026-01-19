# Metrics/cpu/betti0_difference.py

import numpy as np
from skimage.measure import label

from ..base import BaseMetric


class Betti0RelativeError(BaseMetric):
    type = "cpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Relative Betti-0 Characteristic Error (RECE):
        - RECE = 0 means the prediction matches the mask perfectly.
        - The higher the RECE, the greater the difference in connected components or holes between prediction and mask.
        """
        return new_score < old_score

    def __call__(self, prediction: np.ndarray, mask: np.ndarray) -> float:
        """ 
        Relative Betti-0 Characteristic Error (RECE):
        RECE = |E(pred) - E(mask)| / (|E(mask)| + 1e-8)
        where E is the Betti-0 characteristic (number of connected components).
        - RECE = 0 means the prediction matches the mask perfectly.
        - The higher the RECE, the greater the difference in connected components or holes between prediction and mask.
        """
        pred_np = prediction.astype(np.uint8)
        mask_np = mask.astype(np.uint8)
        scores = []
        for i in range(pred_np.shape[0]):
            pred_img = pred_np[i]
            mask_img = mask_np[i]
            n_pred = label(pred_img).max()  # label connected regions of an integer array.
            n_mask = label(
                mask_img).max()  # .max gives us the biggest label index, which corresponds to the number of connected components.
            ratio = (abs(n_pred - n_mask) / n_mask) if n_mask > 0 else (1 if n_pred == 0 else 0)
            scores.append(ratio)
        return float(np.mean(scores))
