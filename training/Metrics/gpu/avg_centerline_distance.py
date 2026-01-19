# Metrics/gpu/centerline_distance.py
import torch

from ..base import BaseMetric


def _to_cu(x: torch.Tensor):
    import cupy as cp
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))


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
        # Fallback CPU (handle both 2D and 3D inputs)
        import numpy as np, cupy as cp
        from skimage.morphology import skeletonize
        if bin_cp.ndim == 2:
            sk = skeletonize(cp.asnumpy(bin_cp).astype(bool))
            return cp.asarray(sk)
        elif bin_cp.ndim == 3:
            sk = [skeletonize(cp.asnumpy(bin_cp[i]).astype(bool)) for i in range(bin_cp.shape[0])]
            return cp.asarray(np.stack(sk, axis=0))
        else:
            raise ValueError(f"Unsupported ndim for skeletonization fallback: {bin_cp.ndim}")


def _edt_to_ones(bin_cp, sampling=None):
    # distance aux "1" via EDT sur l'inverse
    from cupyx.scipy.ndimage import distance_transform_edt
    return distance_transform_edt(1 - bin_cp.astype(bool), sampling=sampling)


class AverageCenterlineDistance(BaseMetric):
    type = "gpu"

    def __init__(self, threshold: float = 0.5, sampling=None):
        super().__init__()
        self.threshold = threshold
        self.sampling = sampling

    def is_better(self, old_score, new_score) -> bool:
        return new_score < old_score

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor):
        import cupy as cp
        import numpy as np
        pred = _ensure_2d_bin(prediction, self.threshold)
        gt = _ensure_2d_bin(mask, self.threshold)
        if not pred.is_cuda or not gt.is_cuda:
            raise ValueError("CenterlineDistance attend des tenseurs CUDA.")
        cp_pred = _to_cu(pred)
        cp_gt = _to_cu(gt)
    
        acd = []
        for i in range(cp_pred.shape[0]):
            sk_pred = _skeletonize_gpu(cp_pred[i]).astype(cp.uint8)
            sk_gt = _skeletonize_gpu(cp_gt[i]).astype(cp.uint8)

            dt_gt = _edt_to_ones(sk_gt, sampling=self.sampling)
            dt_pr = _edt_to_ones(sk_pred, sampling=self.sampling)

            d1 = dt_gt[sk_pred.astype(bool)]  # image with skeletonized prediction (binary)
            d2 = dt_pr[sk_gt.astype(bool)]
            if d1.size == 0 and d2.size == 0:
                continue  # skip empty case
            all_d = cp.concatenate([d1, d2]) if (d1.size and d2.size) else (d1 if d1.size else d2)
            acd.append(float(all_d.mean().get()))
        #print(f"ACD batch: {acd}")
        return float(np.mean(acd)) if acd else float("nan")
