# Metrics/gpu/centerline_distance.py
import torch

from ..base import BaseMetric


def _to_cu(x: torch.Tensor):
    import cupy as cp
    return cp.fromDlpack(torch.utils.dlpack.to_dlpack(x))


def _ensure_2d_bin(t: torch.Tensor, thr: float):
    if t.ndim == 4 and t.shape[1] == 1:
        t = t[:, 0]
    elif t.ndim == 4 and t.shape[1] > 1:
        raise ValueError("Mask/pred doivent être mono-classe (C=1).")
    return (t >= thr).to(torch.uint8)


def _skeletonize_gpu(bin_cp):
    try:
        from cucim.skimage.morphology import thin
        return thin(bin_cp)
    except Exception:
        # Fallback CPU
        import numpy as np, cupy as cp
        from skimage.morphology import skeletonize
        sk = [skeletonize(cp.asnumpy(bin_cp[i]).astype(bool)) for i in range(bin_cp.shape[0])]
        return cp.asarray(np.stack(sk, axis=0))


def _edt_to_ones(bin_cp, sampling=None):
    # distance aux "1" via EDT sur l'inverse
    from cupyx.scipy.ndimage import distance_transform_edt
    return distance_transform_edt(1 - bin_cp.astype(bool), sampling=sampling)


class CenterlineDistance(BaseMetric):
    """
    Retourne un dict {'acd': float, 'hd95': float}
    """
    type = "gpu"

    def __init__(self, threshold: float = 0.5, sampling=None):
        super().__init__()
        self.threshold = threshold
        self.sampling = sampling

    def is_better(self, old_score, new_score) -> bool:
        # on juge sur ACD (plus petit est meilleur)
        return new_score["acd"] < old_score["acd"]

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor):
        import cupy as cp
        pred = _ensure_2d_bin(prediction, self.threshold)
        gt = _ensure_2d_bin(mask, self.threshold)
        if not pred.is_cuda or not gt.is_cuda:
            raise ValueError("CenterlineDistance attend des tenseurs CUDA.")
        cp_pred = _to_cu(pred)
        cp_gt = _to_cu(gt)

        sk_pred = _skeletonize_gpu(cp_pred).astype(cp.uint8)
        sk_gt = _skeletonize_gpu(cp_gt).astype(cp.uint8)

        dt_gt = _edt_to_ones(sk_gt, sampling=self.sampling)
        dt_pr = _edt_to_ones(sk_pred, sampling=self.sampling)

        d1 = dt_gt[sk_pred.astype(bool)]
        d2 = dt_pr[sk_gt.astype(bool)]

        if d1.size == 0 and d2.size == 0:
            return {"acd": float("inf"), "hd95": float("inf")}

        import numpy as np
        all_d = cp.concatenate([d1, d2]) if (d1.size and d2.size) else (d1 if d1.size else d2)
        hd95 = float(cp.percentile(all_d, 95).get())
        return hd95
