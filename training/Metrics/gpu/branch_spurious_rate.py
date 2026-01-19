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
        import numpy as np, cupy as cp
        from skimage.morphology import skeletonize
        sk = [skeletonize(cp.asnumpy(bin_cp[i]).astype(bool)) for i in range(bin_cp.shape[0])]
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


class BranchMetrics(BaseMetric):
    """
    Retourne:
      {
        'break_rate_per_kpx': float,
        'spurious_branch_rate_per_kpx': float,
        'num_bifurcations_pred': float,
        'total_length_pred_px': float
      }
    """
    type = "gpu"

    def __init__(self, threshold: float = 0.5, tol_radius: float = 2.0, min_branch_len: int = 10):
        super().__init__()
        self.threshold = threshold
        self.tol_radius = tol_radius
        self.min_branch_len = int(min_branch_len)

    def is_better(self, old, new) -> bool:
        # on préfère moins de cassures et branches parasites
        score_old = old['break_rate_per_kpx'] + old['spurious_branch_rate_per_kpx']
        score_new = new['break_rate_per_kpx'] + new['spurious_branch_rate_per_kpx']
        return score_new < score_old

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor):
        import cupy as cp
        from cucim.skimage.measure import label

        pred = _ensure_2d_bin(prediction, self.threshold)
        gt = _ensure_2d_bin(mask, self.threshold)
        if not pred.is_cuda or not gt.is_cuda:
            raise ValueError("BranchMetrics attend des tenseurs CUDA.")
        cp_pred = _to_cu(pred)
        cp_gt = _to_cu(gt)

        sk_pred = _skeletonize_gpu(cp_pred).astype(cp.uint8)
        sk_gt = _skeletonize_gpu(cp_gt).astype(cp.uint8)

        # longueur totale (px)
        len_gt = float(sk_gt.sum().get())

        # branches parasites: squelette préd non soutenu par GT (hors tolérance)
        extra = sk_pred.astype(bool) & (~_binary_dilation(sk_gt, radius=self.tol_radius))
        lab, ncomp = label(extra.astype(cp.uint8), return_num=True)
        if ncomp > 0:
            sizes = cp.bincount(lab.ravel())
            # bin 0 = fond
            valid = (cp.arange(sizes.size) > 0) & (sizes >= self.min_branch_len)
            spurious = int(valid.sum().get())
        else:
            spurious = 0
        spurious_rate = (spurious / len_gt) if len_gt > 0 else 0.0

        return float(spurious_rate)
