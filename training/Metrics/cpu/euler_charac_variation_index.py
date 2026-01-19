# Metrics/cpu/euler_charac_difference.py

import numpy as np
from skimage.measure import euler_number

from ..base import BaseMetric


class EulerCharacVariationIndex(BaseMetric):
    type = "cpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Variation of Euler Characteristic (VEC):
        - VEC = 0 means the prediction matches the mask perfectly.
        - VEC > 0 means the prediction has more connected components or holes than the mask.
        - VEC < 0 means the prediction has fewer connected components or holes than the mask.
        """
        return abs(new_score) <= abs(old_score)

    def __call__(self, prediction: np.ndarray, mask: np.ndarray) -> float:
        """ 
        Variation of Euler Characteristic (VEC) :
        VEC = (E(pred) - E(mask)) / (E(pred) + E(mask))
        where E is the Euler characteristic.
        
        - VEC = 0 means the prediction matches the mask perfectly.
        - VEC > 0 means the prediction has more connected components or holes than the mask.
        - VEC < 0 means the prediction has fewer connected components or holes than the mask.
        """
        pred_np = prediction.astype(np.uint8)
        mask_np = mask.astype(np.uint8)
        batch_size = pred_np.shape[0]
        scores = []
        for i in range(batch_size):
            pred_img = pred_np[i]
            mask_img = mask_np[i]
            e_pred = euler_number(pred_img, connectivity=1)  # number of connected components - number of holes
            e_mask = euler_number(mask_img, connectivity=1)
            ratio = abs(e_pred - e_mask) / (e_mask + e_pred) if e_mask > 0 else (1 if e_pred == 0 else 0)
            scores.append(ratio)
        return float(np.mean(scores))
