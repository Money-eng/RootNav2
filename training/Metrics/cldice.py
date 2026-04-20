import numpy as np
from skimage.morphology import skeletonize
import torch


from .base import BaseMetric

def cl_score(v, s):
    sum_s = np.sum(s)
    if sum_s == 0: # if skeleton is empty, return 0 to avoid division by zero
        return 0.0
    return float(np.sum(v * s)) / float(sum_s) # always between ]0, 1]

def clDice(v_p, v_l):
    v_p_bool = v_p > 0
    v_l_bool = v_l > 0
    
    skel_p = skeletonize(v_p_bool)
    skel_l = skeletonize(v_l_bool)

    tprec = cl_score(v_p_bool, skel_l) # between 0 and 1
    tsens = cl_score(v_l_bool, skel_p) # between 0 and 1

    if (tprec + tsens) == 0:
        return 0.0

    return 2.0 * tprec * tsens / (tprec + tsens)

class CLDice(BaseMetric):
    type = "gpu" # ish

    def __init__(self):
        super().__init__()

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = prediction.detach().cpu().numpy()
        msk = mask.detach().cpu().numpy()

        
        cldice_scores = []
        
        for i in range(pred.shape[0]):
            p = np.squeeze(pred[i])
            m = np.squeeze(msk[i])
            
            score = clDice(p, m)
            cldice_scores.append(score)
            
        return float(np.mean(cldice_scores))