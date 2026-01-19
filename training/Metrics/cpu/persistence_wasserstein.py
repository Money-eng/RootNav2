import gudhi as gd
import numpy as np
from gudhi.wasserstein import wasserstein_distance
from scipy.ndimage import distance_transform_edt


class PeristenceWasserstein():
    type = "cpu"

    def __init__(
            self,
            homology_dimensions=(0, 1),
            complex_builder=None,
            filter_fn=None,
    ):
        """
        - homology_dimensions : tuple des dimensions d'homologie à considérer (ex. (0,1)).
        - complex_builder : fonction data -> complexe GUDHI  
            (ex. lambda data: gd.CubicalComplex(top_dimensional_cells=data))
        - filter_fn : fonction data -> data filtrée  
            (ex. pour superlevel sets, lambda d: d.max() - d)
        """
        super().__init__()
        self.homology_dimensions = homology_dimensions
        # builder par défaut : CubicalComplex en sous-niveau
        self.complex_builder = complex_builder or (
            lambda data: gd.CubicalComplex(top_dimensional_cells=data)
        )
        # filter par défaut : pas de transformation
        self.filter_fn = filter_fn or (
            lambda data: distance_transform_edt(data > 0)
        )

    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Bottleneck distance : 
        - plus la distance est petite, plus la prédiction est proche du masque.
        - distance = 0 signifie que les diagrammes sont identiques.
        """
        return new_score < old_score

    def _compute_diagram(self, array: np.ndarray):
        # applique le filtre puis construit le complexe et calcule la persistance
        data = self.filter_fn(array.astype(np.float32))
        complex_ = self.complex_builder(data)
        complex_.compute_persistence()
        return complex_.persistence()

    def __call__(self, prediction: np.ndarray, mask: np.ndarray):
        # B, C, H, W -> treating each image independently
        distances = []
        # pred = prediction.squeeze(1)
        # msk = mask.squeeze(1) # remove channel dimension if present
        for i in range(prediction.shape[0]):
            # calcule diagrammes
            diag_pred = self._compute_diagram(prediction[i])
            diag_msk = self._compute_diagram(mask[i])
            distance_per_dim = {}
            # Bottleneck par dimension
            for dim in self.homology_dimensions:
                dgm_pred = [(b, d) for dgm_dim, (b, d) in diag_pred if dgm_dim == dim]
                dgm_msk = [(b, d) for dgm_dim, (b, d) in diag_msk if dgm_dim == dim]
                distance_per_dim[dim] = wasserstein_distance(
                    dgm_pred, dgm_msk
                )
            # ignore if NA 
            if any(np.isnan(list(distance_per_dim.values()))):
                continue
            distances.append(distance_per_dim)
        # mean for homology dim 0 and 1 (cc and loops)
        mean_distances = {
            dim: np.mean([d[dim] for d in distances]) for dim in self.homology_dimensions
        }
        return mean_distances
