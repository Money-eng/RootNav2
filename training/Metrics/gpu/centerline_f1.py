# Metrics/gpu/centerline_f1.py
import cupy as cp
import torch

from ..base import BaseMetric

# cache flag to disable cucim after first failure
_USE_CUCIM = True


def _to_cu(x: torch.Tensor):
    # ensure on cuda then zero-copy via dlpack
    if not x.is_cuda:
        x = x.to("cuda", non_blocking=True)
    return cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))


def _ensure_2d_bin(t: torch.Tensor, thr: float):
    # Accept (N,1,H,W) or (N,H,W)
    if t.ndim == 4 and t.shape[1] == 1:
        t = t[:, 0]
    elif t.ndim == 4 and t.shape[1] > 1:
        raise ValueError("Mask/pred doivent être mono-classe (C=1).")
    return (t >= thr).to(torch.uint8)


def _skeletonize_gpu(bin_cp: cp.ndarray):
    """
    bin_cp: 2D cupy array (single sample)
    Returns cupy uint8 skeleton.
    """
    global _USE_CUCIM
    # Try cucim once unless already disabled
    if _USE_CUCIM:
        try:
            from cucim.skimage.morphology import thin
            sk = thin(bin_cp.astype(cp.uint8))
            return (sk > 0).astype(cp.uint8)
        except Exception:
            _USE_CUCIM = False  # disable for subsequent calls

    # CPU fallback (scikit-image)
    from skimage.morphology import skeletonize
    bin_np = cp.asnumpy(bin_cp).astype(bool)
    sk_np = skeletonize(bin_np)
    return cp.asarray(sk_np.astype("uint8"))


def _edt_gpu(bin_cp, sampling=None):
    from cupyx.scipy.ndimage import distance_transform_edt
    inv = 1 - bin_cp
    return distance_transform_edt(inv, sampling=sampling)


def _precision_recall_f1(pred_skel, gt_skel, r, sampling=None):
    dt_gt = _edt_gpu(gt_skel.astype(cp.uint8), sampling=sampling)
    dt_pr = _edt_gpu(pred_skel.astype(cp.uint8), sampling=sampling)

    pred_pts = pred_skel.astype(bool)
    gt_pts = gt_skel.astype(bool)

    d_pred_to_gt = dt_gt[pred_pts]
    d_gt_to_pred = dt_pr[gt_pts]

    precision = float((d_pred_to_gt <= r).mean()) if d_pred_to_gt.size else 0.0
    recall = float((d_gt_to_pred <= r).mean()) if d_gt_to_pred.size else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1


class CenterlineF1(BaseMetric):
    type = "gpu"

    def __init__(self, threshold: float = 0.5, radius: float = 5.0, sampling=None):
        super().__init__()
        self.threshold = threshold
        self.radius = radius
        self.sampling = sampling

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score > old_score

    @torch.no_grad()
    def __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float:
        pred = _ensure_2d_bin(prediction, self.threshold).contiguous()
        gt = _ensure_2d_bin(mask, self.threshold).contiguous()

        if not pred.is_cuda or not gt.is_cuda:
            raise ValueError("CenterlineF1 attend des tenseurs sur GPU (CUDA).")

        cp_pred = _to_cu(pred)
        cp_gt = _to_cu(gt)

        f1_list = []
        # Iterate batch
        for i in range(cp_pred.shape[0]):
            sk_pred = _skeletonize_gpu(cp_pred[i])
            sk_gt = _skeletonize_gpu(cp_gt[i])
            _, _, f1 = _precision_recall_f1(sk_pred, sk_gt, self.radius, self.sampling)
            f1_list.append(f1)

        return float(sum(f1_list) / len(f1_list)) if f1_list else 0.0
