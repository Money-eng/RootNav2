import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

from ..base import BaseMetric


class VI(BaseMetric):
    type = "cpu"

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Variation of Information (VI):
        - VI = 0 means the prediction matches the mask perfectly.
        - The lower the VI, the closer the prediction is to the mask.
        """
        return new_score < old_score

    def __call__(self, prediction: np.ndarray, mask: np.ndarray) -> float:
        # Aplatir les masques
        pred = prediction.flatten()
        mk = mask.flatten()

        # Histogrammes
        _, u_counts = np.unique(pred, return_counts=True)
        _, v_counts = np.unique(mk, return_counts=True)

        p_u = u_counts / np.sum(u_counts)
        p_v = v_counts / np.sum(v_counts)

        # Entropies
        H_u = entropy(p_u)
        H_v = entropy(p_v)

        # Mutual Information
        mi = mutual_info_score(pred, mk)

        # Variation of Information
        return H_u + H_v - 2 * mi
