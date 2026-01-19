# Metrics/cpu/betti0_difference.py

import numpy as np
from skimage.measure import label

from ..base import BaseMetric


class Betti0VariationIndex(BaseMetric):
    type = "cpu"

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

    def __call__(self, prediction: np.ndarray, mask: np.ndarray) -> float:
        """ 
        Variation of Betti-0 Characteristic (VEC):
        VEC = (E(pred) - E(mask)) / (E(pred) + E(mask))
        where E is the Betti-0 characteristic (number of connected components).
        
        - VEC = 0 means the prediction matches the mask perfectly.
        - VEC > 0 means the prediction has more connected components or holes than the mask.
        - VEC < 0 means the prediction has fewer connected components or holes than the mask.
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
            ratio = (abs(n_pred - n_mask) / (n_mask + n_pred)) if n_mask > 0 else (1 if n_pred == 0 else 0)
            scores.append(ratio)
        return float(np.mean(scores))
