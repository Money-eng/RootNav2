# PersistenceBottleneckParallel: EDT sur GPU (CuPy) + persistance & bottleneck en CPU parallélisé (GUDHI).
# API: callable(prediction, mask) -> {dim: mean_bottleneck_distance}

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, Optional, Tuple, List

import numpy as np

# --- dépendances optionnelles
try:
    import torch
except Exception:
    torch = None

try:
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as edt_gpu

    _HAS_CUPY = True
except Exception:
    _HAS_CUPY = False

try:
    from scipy.ndimage import distance_transform_edt as edt_cpu
except Exception:
    edt_cpu = None

import gudhi as gd


# ---------- Helpers picklables (top-level) ----------

def _build_persistence_and_bottleneck(
        filt_pred_np: np.ndarray,
        filt_mask_np: np.ndarray,
        homology_dims: Tuple[int, ...],
) -> Dict[int, float]:
    """Worker CPU: construit les complexes, calcule la persistance et les distances bottleneck."""
    cc_pred = gd.CubicalComplex(top_dimensional_cells=filt_pred_np.astype(np.float32, copy=False))
    cc_pred.compute_persistence()
    pers_pred = cc_pred.persistence()

    cc_mask = gd.CubicalComplex(top_dimensional_cells=filt_mask_np.astype(np.float32, copy=False))
    cc_mask.compute_persistence()
    pers_mask = cc_mask.persistence()

    out = {}
    for dim in homology_dims:
        dgm_pred = [(b, d) for d, (b, d) in pers_pred if d == dim]
        dgm_mask = [(b, d) for d, (b, d) in pers_mask if d == dim]
        out[dim] = gd.bottleneck_distance(dgm_pred, dgm_mask)
    return out


def _maybe_squeeze_channel(x, expect_3d=True):
    # attend (B,H,W) ou (B,1,H,W) -> renvoie (B,H,W)
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0]
    if x.ndim == 3:
        return x
    if expect_3d:
        raise ValueError(f"Shape attendu (B,H,W) ou (B,1,H,W), reçu {x.shape}")
    return x


def _to_cupy(x) -> "cp.ndarray":
    if not _HAS_CUPY:
        raise RuntimeError("CuPy indisponible : installez cupy-cudaXX ou utilisez le fallback CPU.")
    if isinstance(x, np.ndarray):
        return cp.asarray(x)  # host->device
    if torch is not None and torch.is_tensor(x):
        if x.is_cuda:
            return cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
        return cp.asarray(x.detach().cpu().numpy())
    if _HAS_CUPY and isinstance(x, cp.ndarray):
        return x
    raise TypeError(f"Type non supporté pour conversion vers CuPy: {type(x)}")


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if _HAS_CUPY and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    raise TypeError(f"Type non supporté pour conversion vers NumPy: {type(x)}")


# ---------- Classe principale ----------

class PersistenceBottleneckGPUParallel:
    type = "gpu"

    def __init__(
            self,
            homology_dimensions: Iterable[int] = (0, 1),
            use_gpu_filter: bool = True,
            max_workers: Optional[int] = None,
            binarize_threshold: Optional[float] = 0.0,
            edt_on_foreground: bool = True,
    ):
        """
        - homology_dimensions : dimensions d'homologie à comparer (ex: (0,1)).
        - use_gpu_filter : si True et CuPy dispo -> EDT sur GPU, sinon fallback CPU.
        - max_workers : nombre de workers CPU (par défaut: os.cpu_count()).
        - binarize_threshold : si non None, binarise (x > t) avant EDT. Si None, utilise tel quel.
        - edt_on_foreground : EDT sur le foreground (>0). Si False, EDT sur le background (==0).
        """
        self.homology_dimensions = tuple(homology_dimensions)
        self.use_gpu_filter = bool(use_gpu_filter and _HAS_CUPY)
        self.max_workers = max_workers or os.cpu_count() or 1
        self.binarize_threshold = binarize_threshold
        self.edt_on_foreground = bool(edt_on_foreground)

        if not self.use_gpu_filter and edt_cpu is None:
            raise RuntimeError("Ni CuPy ni SciPy EDT disponibles : installez l'un des deux.")

    def is_better(self, old_score: float, new_score: float) -> bool:
        return new_score < old_score

    def _gpu_filter(self, arr_cp: "cp.ndarray") -> "cp.ndarray":
        """Filtration par défaut: EDT sur GPU (CuPy). Retour float32 HxW."""
        a = arr_cp
        if self.binarize_threshold is not None:
            a = (a > self.binarize_threshold).astype(cp.uint8)
        else:
            # si déjà binaire, ok ; sinon seuil léger sur >0 par défaut
            a = (a > 0).astype(cp.uint8)
        if not self.edt_on_foreground:
            a = (a == 0).astype(cp.uint8)
        return edt_gpu(a).astype(cp.float32)

    def _cpu_filter(self, arr_np: np.ndarray) -> np.ndarray:
        """Fallback filtration CPU (SciPy EDT)."""
        a = arr_np
        if self.binarize_threshold is not None:
            a = (a > self.binarize_threshold).astype(np.uint8)
        else:
            a = (a > 0).astype(np.uint8)
        if not self.edt_on_foreground:
            a = (a == 0).astype(np.uint8)
        return edt_cpu(a).astype(np.float32)

    def _prepare_filtrations_pair(
            self, pred_slice, mask_slice
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule la filtration (EDT) pour pred et mask, idéalement sur GPU,
        puis renvoie des numpy (CPU) float32 prêts pour GUDHI.
        """
        if self.use_gpu_filter:
            pred_cp = _to_cupy(pred_slice)
            mask_cp = _to_cupy(mask_slice)
            f_pred_cp = self._gpu_filter(pred_cp)
            f_mask_cp = self._gpu_filter(mask_cp)
            f_pred_np = _to_numpy(f_pred_cp)
            f_mask_np = _to_numpy(f_mask_cp)
            return f_pred_np, f_mask_np
        else:
            pred_np = _to_numpy(pred_slice)
            mask_np = _to_numpy(mask_slice)
            f_pred_np = self._cpu_filter(pred_np)
            f_mask_np = self._cpu_filter(mask_np)
            return f_pred_np, f_mask_np

    def __call__(self, prediction, mask) -> Dict[int, float]:
        """
        prediction, mask : (B,H,W) ou (B,1,H,W) ; numpy ou torch.Tensor (CPU/GPU).
        Retour: {dim: mean_bottleneck_distance}
        """
        # convert/validate shapes
        pred_any = prediction
        mask_any = mask

        # On ne touche pas encore au type (CuPy/NumPy/Torch) : on travaille slice par slice
        # pour limiter la mémoire GPU.
        if hasattr(pred_any, "shape"):
            B = pred_any.shape[0]
        else:
            raise ValueError("prediction doit avoir un attribut shape")

        # squeeze canal si nécessaire (sur le type d'origine)
        if _HAS_CUPY and isinstance(pred_any, cp.ndarray):
            pred_any = _maybe_squeeze_channel(pred_any)
            mask_any = _maybe_squeeze_channel(mask_any)
        else:
            pred_any = _maybe_squeeze_channel(pred_any)
            mask_any = _maybe_squeeze_channel(mask_any)

        # Soumissions parallèles CPU: chaque tâche reçoit les deux filtrations en numpy
        results: List[Dict[int, float]] = []
        futures = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            for i in range(B):
                pred_i = pred_any[i]
                mask_i = mask_any[i]
                f_pred_np, f_mask_np = self._prepare_filtrations_pair(pred_i, mask_i)
                futures.append(
                    ex.submit(
                        _build_persistence_and_bottleneck,
                        f_pred_np, f_mask_np, self.homology_dimensions
                    )
                )
            for fut in as_completed(futures):
                dist_dim = fut.result()
                if not any(np.isnan(list(dist_dim.values()))):
                    results.append(dist_dim)

        if not results:
            return {dim: float("nan") for dim in self.homology_dimensions}

        # moyenne par dimension
        out = {
            dim: float(np.mean([d[dim] for d in results]))
            for dim in self.homology_dimensions
        }
        return out
