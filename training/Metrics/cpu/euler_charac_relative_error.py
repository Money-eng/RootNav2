# Metrics/cpu/euler_charac_difference.py

import numpy as np
from skimage.measure import euler_number

from ..base import BaseMetric


class EulerCharacRelativeError(BaseMetric):
    type = "cpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Relative Euler Characteristic Error (RECE) :
        - RECE = 0 means the prediction matches the mask perfectly.
        - The higher the RECE, the greater the difference in connected components or holes between prediction and mask.
        """
        return new_score < old_score

    def __call__(self, prediction: np.ndarray, mask: np.ndarray) -> float:
        """ 
        Relative Euler Characteristic Error (RECE) :
        RECE = |E(pred) - E(mask)| / (|E(mask)| + 1e-8)
        where E is the Euler characteristic.
        - RECE = 0 means the prediction matches the mask perfectly.
        - The higher the RECE, the greater the difference in connected components or holes between prediction and mask.
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
            diff = abs(e_pred - e_mask) / (abs(e_mask) + 1e-8)  # Avoid division by zero
            scores.append(diff)
        return float(np.mean(scores))
