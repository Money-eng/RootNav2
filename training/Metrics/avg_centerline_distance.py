# Metrics/gpu/centerline_distance.py
import torch
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt


from .base import BaseMetric

def compute_centerline_metric(y_pred, y_true, pixel_size=0.076):
    pred_sum = np.count_nonzero(y_pred)
    true_sum = np.count_nonzero(y_true)

    if pred_sum == 0 and true_sum == 0:
        return 0.0
    elif pred_sum == 0 or true_sum == 0:
        return float('inf')

    skel_pred = skeletonize(y_pred.astype(bool))
    skel_gt = skeletonize(y_true.astype(bool))

    dist_map_gt = distance_transform_edt(np.logical_not(skel_gt))
    dist_map_pred = distance_transform_edt(np.logical_not(skel_pred))

    d1 = dist_map_gt[skel_pred]
    d2 = dist_map_pred[skel_gt]
    
    d1 = d1 * pixel_size
    d2 = d2 * pixel_size
    
    if d1.size == 0 and d2.size == 0:
        return 0.0 

    all_d = np.concatenate([d1, d2])
    return float(np.mean(all_d))

class AverageSymetricCenterlineDistance(BaseMetric):
    type = "cpu"

    def __init__(self, sampling=None):
        super().__init__()
        self.sampling = sampling
        self.pixel_size = 76 * 1e-3 

    def is_better(self, old_score, new_score) -> bool:
        return new_score < old_score

    @torch.no_grad()
    def __call__(self, prediction: np.ndarray, mask: np.ndarray) -> float:
        pred_np = np.squeeze(prediction.cpu().numpy()).astype(np.uint8)
        gt_np = np.squeeze(mask.cpu().numpy()).astype(np.uint8)
        return compute_centerline_metric(pred_np, gt_np, pixel_size=self.pixel_size)
