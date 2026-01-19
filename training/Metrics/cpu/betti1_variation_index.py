# Metrics/cpu/betti0_difference.py

import numpy as np
from skimage.measure import label

from ..base import BaseMetric


class Betti1VariationIndex(BaseMetric):
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
            n_pred = label(~pred_img.astype(bool)).max()
            n_mask = label(~mask_img.astype(bool)).max()
            b1_pred = max(0, n_pred - 1)  # Betti-1 = number of holes = cc_inverted - 1
            b1_mask = max(0, n_mask - 1)
            jaccard_ratio = (abs(b1_pred - b1_mask) / (b1_mask + b1_pred)) if b1_mask > 0 else (
                1 if b1_pred == 0 else 0)
            scores.append(jaccard_ratio)
        return float(np.mean(scores))
