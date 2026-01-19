# Metrics/cpu/betti0_difference.py

import numpy as np
from skimage.measure import label

from ..base import BaseMetric


class Betti0JaccardRatio(BaseMetric):
    type = "cpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Jaccard Ratio of Betti-0 Characteristic (JREC) :
        - JREC = 1 means the prediction matches the mask perfectly.
        - The lower the JREC, the greater the difference in connected components between prediction and mask.
        """
        return new_score > old_score

    def __call__(self, prediction: np.ndarray, mask: np.ndarray) -> float:
        """
        Jaccard Ratio of Betti-0 Characteristic (JREC) :
        JREC = min(|E(pred)|, |E(mask)|) / (max(|E(pred)|, |E(mask)|) + 1e-8)
        where E is the Betti-0 characteristic (number of connected components).
        - JREC = 1 means the prediction matches the mask perfectly.
        - The lower the JREC, the greater the difference in connected components between prediction and mask.
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
            jaccard_ratio = (min(n_pred, n_mask) / (max(n_pred, n_mask) + 1e-8))  # Avoid division by zero
            scores.append(jaccard_ratio)
        return float(np.mean(scores))
