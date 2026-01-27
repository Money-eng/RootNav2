# Metrics/__init__.py
from .base import BaseMetric
from .cpu.betti0_ratio import Betti0JaccardRatio
from .cpu.betti0_relative_error import Betti0RelativeError
from .cpu.betti0_variation_index import Betti0VariationIndex
from .cpu.betti1_ratio import Betti1JaccardRatio
from .cpu.betti1_relative_error import Betti1RelativeError
from .cpu.betti1_variation_index import Betti1VariationIndex
from .cpu.euler_charac_abs_ratio import EulerCharaJaccardsRatio
from .cpu.euler_charac_relative_error import EulerCharacRelativeError
from .cpu.euler_charac_variation_index import EulerCharacVariationIndex
# from .cpu.persistence_bottleneck import PeristenceBottleneck
# from .cpu.persistence_wasserstein import PeristenceWasserstein
from .cpu.variation_of_information import VI
from .gpu.avg_centerline_distance import AverageCenterlineDistance
from .gpu.betti0_ratio_gpu import Betti0JaccardRatioGPU
from .gpu.betti0_relative_error_gpu import Betti0RelativeErrorGPU
from .gpu.betti0_variation_index_gpu import Betti0VariationIndexGPU
from .gpu.betti1_ratio_gpu import Betti1JaccardRatioGPU
from .gpu.betti1_relative_error_gpu import Betti1RelativeErrorGPU
from .gpu.betti1_variation_index_gpu import Betti1VariationIndexGPU
from .gpu.centerline_f1 import CenterlineF1
from .gpu.cldice import CLDice
from .gpu.dice import Dice
from .gpu.f1_score import F1Score
from .gpu.focal import FocalLoss
from .gpu.haussdorff import HausdorffDistance
from .gpu.iou import MeanIoU
from .gpu.mutual_information import NormalizedMutualInformation
from .gpu.precision import Precision
from .gpu.recall import Recall

# Global dictionnary to map metric names to their corresponding classes
METRIC_FACTORIES = {
    # GPU
    "dice": Dice,
    "cldice": CLDice,
    "focal_loss": FocalLoss,
    "f1_score": F1Score,
    "mean_iou": MeanIoU,
    "precision": Precision,
    "recall": Recall,
    "hausdorff_distance": HausdorffDistance,
    "normalized_mutual_information": NormalizedMutualInformation,
    "betti0_jaccard_ratio_gpu": Betti0JaccardRatioGPU,
    "betti0_relative_error_gpu": Betti0RelativeErrorGPU,
    "betti0_variation_index_gpu": Betti0VariationIndexGPU,
    "betti1_jaccard_ratio_gpu": Betti1JaccardRatioGPU,
    "betti1_relative_error_gpu": Betti1RelativeErrorGPU,
    "betti1_variation_index_gpu": Betti1VariationIndexGPU,
    "centerline_f1_gpu": CenterlineF1,
    "average_centerline_distance": AverageCenterlineDistance,

    # CPU
    "variation_of_information": VI,
    "betti0_jaccard_ratio": Betti0JaccardRatio,
    "betti0_relative_error": Betti0RelativeError,
    "betti0_variation_index": Betti0VariationIndex,
    "betti1_jaccard_ratio": Betti1JaccardRatio,
    "betti1_relative_error": Betti1RelativeError,
    "betti1_variation_index": Betti1VariationIndex,
    "euler_charac_jaccard_ratio": EulerCharaJaccardsRatio,
    "euler_charac_relative_error": EulerCharacRelativeError,
    "euler_charac_variation_index": EulerCharacVariationIndex,
    # "persistence_bottleneck": PeristenceBottleneck,
    # "persistence_wasserstein": PeristenceWasserstein
}


def get_metric(metric_config: dict) -> BaseMetric:
    """
    Instanciate a given metric based on its configuration.
    {
        "name": "dice",
        "params": {...} 
    }
    """
    name = metric_config["name"]
    params = metric_config.get("params", {})
    if name not in METRIC_FACTORIES:
        raise ValueError(
            f"Unknown metric: {name}. Known: {list(METRIC_FACTORIES.keys())}")
    try:
        return METRIC_FACTORIES[name](**params)
    except TypeError as e:
        raise TypeError(
            f"Error instantiating metric '{name}' with params {params}: {e}")


def get_metrics(metrics_config: dict) -> dict:
    """
    Instanciate all metrics based on the provided configuration.
    The configuration should be a dictionary with two keys: "cpu" and "gpu",
    each containing a list of metric configurations.    
    Example:
    {
        "cpu": [
            {"name": "connectivity", "params": {}},
            {"name": "ari_index", "params": {}}
        ],
        "gpu": [
            {"name": "dice", "params": {}},
            {"name": "iou", "params": {}}
        ]
    }
    """
    result = {"cpu": [], "gpu": [], "mtg": []}
    for t in ["cpu", "gpu", "mtg"]:
        for cfg in metrics_config.get(t, []):
            metric = get_metric(cfg)
            result[t].append(metric)
    return result
