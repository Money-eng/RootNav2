# PersistenceWassersteinParallel: EDT sur GPU (CuPy) + persistance & Wasserstein en CPU parallélisé (GUDHI).
# callable(prediction, mask) -> {dim: mean_wasserstein_distance}

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

# GUDHI a deux chemins possibles pour Wasserstein selon la version
try:
    from gudhi.wasserstein import wasserstein_distance as _gd_wasserstein
except Exception:
    try:
        from gudhi.hera import wasserstein_distance as _gd_wasserstein
    except Exception:
        _gd_wasserstein = None


# ---------- Helpers picklables (top-level) ----------

def _finite_points(arr: np.ndarray) -> np.ndarray:
    """Filtre tout point ayant NaN/inf (prudence selon complexes / filtrations)."""
    if arr.size == 0:
        return arr
    mask = np.isfinite(arr).all(axis=1)
    return arr[mask]


def _build_persistence_and_wasserstein(
        filt_pred_np: np.ndarray,
        filt_mask_np: np.ndarray,
        homology_dims: Tuple[int, ...],
        order: float,
        internal_p: float,
) -> Dict[int, float]:
    """Worker CPU: construit les complexes, calcule la persistance et les distances de Wasserstein."""
    if _gd_wasserstein is None:
        raise RuntimeError("GUDHI 'wasserstein_distance' introuvable (gudhi.wasserstein ou gudhi.hera).")

    # Complexes & persistance
    cc_pred = gd.CubicalComplex(top_dimensional_cells=filt_pred_np.astype(np.float32, copy=False))
    cc_pred.compute_persistence()
    pers_pred = cc_pred.persistence()

    cc_mask = gd.CubicalComplex(top_dimensional_cells=filt_mask_np.astype(np.float32, copy=False))
    cc_mask.compute_persistence()
    pers_mask = cc_mask.persistence()

    out = {}
    for dim in homology_dims:
        pts_pred = np.array([(b, d) for dgm_dim, (b, d) in pers_pred if dgm_dim == dim], dtype=np.float64)
        pts_mask = np.array([(b, d) for dgm_dim, (b, d) in pers_mask if dgm_dim == dim], dtype=np.float64)
        pts_pred = _finite_points(pts_pred)
        pts_mask = _finite_points(pts_mask)
        # La routine GUDHI gère la projection sur la diagonale
        dist = _gd_wasserstein(pts_pred, pts_mask, order=order, internal_p=internal_p)
        out[dim] = float(dist)
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

class PersistenceWassersteinGPUParallel:
    type = "gpu"  # filtration GPU ; distances sur CPU

    def __init__(
            self,
            homology_dimensions: Iterable[int] = (0, 1),
            use_gpu_filter: bool = True,
            max_workers: Optional[int] = None,
            binarize_threshold: Optional[float] = 0.0,
            edt_on_foreground: bool = True,
            order: float = 1.0,  # W_q : q=1 ou 2 le plus courant
            internal_p: float = 2.0,  # métrique interne: L2 par défaut
    ):
        """
        - homology_dimensions : dimensions d'homologie (0 -> H0, 1 -> H1, etc).
        - use_gpu_filter : True -> EDT CuPy si dispo, sinon fallback SciPy.
        - max_workers : nb de workers CPU (par défaut: os.cpu_count()).
        - binarize_threshold : si non None, binarise (x > t) avant EDT.
        - edt_on_foreground : EDT sur le foreground (>0). Sinon sur background.
        - order : q de la distance de Wasserstein (1 ou 2 typiquement).
        - internal_p : p de la métrique interne (L2=2, L1=1, L∞=np.inf).
        """
        self.homology_dimensions = tuple(homology_dimensions)
        self.use_gpu_filter = bool(use_gpu_filter and _HAS_CUPY)
        self.max_workers = max_workers or os.cpu_count() or 1
        self.binarize_threshold = binarize_threshold
        self.edt_on_foreground = bool(edt_on_foreground)
        self.order = float(order)
        self.internal_p = float(internal_p)

        if _gd_wasserstein is None:
            raise RuntimeError("GUDHI 'wasserstein_distance' introuvable : installez une version avec ce module.")
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
        Retour: {dim: mean_wasserstein_distance}
        """
        if hasattr(prediction, "shape"):
            B = prediction.shape[0]
        else:
            raise ValueError("prediction doit avoir un attribut shape")

        # squeeze canal si nécessaire
        prediction = _maybe_squeeze_channel(prediction)
        mask = _maybe_squeeze_channel(mask)

        results: List[Dict[int, float]] = []
        futures = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            for i in range(B):
                pred_i = prediction[i]
                mask_i = mask[i]
                f_pred_np, f_mask_np = self._prepare_filtrations_pair(pred_i, mask_i)
                futures.append(
                    ex.submit(
                        _build_persistence_and_wasserstein,
                        f_pred_np, f_mask_np,
                        self.homology_dimensions,
                        self.order, self.internal_p
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
