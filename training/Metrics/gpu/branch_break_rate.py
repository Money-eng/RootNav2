# Metrics/gpu/branch_metrics.py
import torch

from ..base import BaseMetric


def _to_cu(x: torch.Tensor):
    import cupy as cp
    return cp.fromDlpack(torch.utils.dlpack.to_dlpack(x))


def _ensure_2d_bin(t: torch.Tensor, thr: float):
    if t.ndim == 4 and t.shape[1] == 1:
        t = t[:, 0]
    elif t.ndim == 4 and t.shape[1] > 1:
        raise ValueError("Mask/pred mono-classe requis (C=1).")
    return (t >= thr).to(torch.uint8)


def _skeletonize_gpu(bin_cp):
    try:
        from cucim.skimage.morphology import thin
        return thin(bin_cp)
    except Exception:
        import numpy as np
        import cupy as cp
        from skimage.morphology import skeletonize
        sk = [skeletonize(cp.asnumpy(bin_cp[i]).astype(bool))
              for i in range(bin_cp.shape[0])]
        return cp.asarray(np.stack(sk, axis=0))


def _neighbors_count(skel):
    # convolution 3x3 pour degré 8-connexe
    from cupyx.scipy.ndimage import convolve
    import cupy as cp
    k = cp.ones((3, 3), dtype=cp.int32)
    k[1, 1] = 0
    nb = convolve(skel.astype(cp.int32), k, mode='constant', cval=0)
    return nb


def _binary_dilation(x, radius=1):
    import cupy as cp
    from cupyx.scipy.ndimage import binary_dilation
    # disque binaire (approx) de rayon r
    r = int(max(1, round(radius)))
    yy, xx = cp.ogrid[-r:r + 1, -r:r + 1]
    se = (yy * yy + xx * xx) <= r * r
    return binary_dilation(x.astype(bool), structure=se)


class BranchBrakeRate(BaseMetric):
    type = "gpu"

    def __init__(self, threshold: float = 0.5, tol_radius: float = 2.0, min_branch_len: int = 10):
        super().__init__()
        self.threshold = threshold
        self.tol_radius = tol_radius
        self.min_branch_len = int(min_branch_len)

    def is_better(self, old, new) -> bool:
        return new < old

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor):
        import cupy as cp

        pred = _ensure_2d_bin(prediction, self.threshold)
        gt = _ensure_2d_bin(mask, self.threshold)
        if not pred.is_cuda or not gt.is_cuda:
            raise ValueError("BranchBrakeRate attend des tenseurs CUDA.")
        cp_pred = _to_cu(pred)
        cp_gt = _to_cu(gt)

        sk_pred = _skeletonize_gpu(cp_pred).astype(cp.uint8)
        sk_gt = _skeletonize_gpu(cp_gt).astype(cp.uint8)

        # longueur totale (px)
        len_gt = float(sk_gt.sum().get())

        # degrés pour endpoints/junctions
        deg_gt = _neighbors_count(sk_gt)
        endpoints_gt = int(((sk_gt == 1) & (deg_gt == 1)).sum().get())

        # cassures: couvrir le squelette GT par une dilatation de la prédiction
        cover = _binary_dilation(
            sk_pred, radius=self.tol_radius) & sk_gt.astype(bool)
        deg_cover = _neighbors_count(cover.astype(cp.uint8))
        endpoints_cover = int(((cover == 1) & (deg_cover == 1)).sum().get())
        # approx: chaque cassure crée ~2 endpoints supplémentaires
        breaks = max(0, (endpoints_cover - endpoints_gt) // 2)
        break_rate = (breaks / len_gt) if len_gt > 0 else 0.0

        return float(break_rate)
