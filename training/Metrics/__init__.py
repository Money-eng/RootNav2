# Metrics/__init__.py
from .base import BaseMetric
from .avg_centerline_distance import AverageSymetricCenterlineDistance
from .betti0_ratio_gpu import Betti0JaccardRatioGPU
from .betti0_variation_index_gpu import Betti0VariationIndexGPU
from .betti1_ratio_gpu import Betti1JaccardRatioGPU
from .betti1_variation_index_gpu import Betti1VariationIndexGPU
from .cldice import CLDice
from .dice import Dice
from .f1_score import F1Score
from .focal import FocalLoss
from .haussdorff import HausdorffDistance
from .haussdorff_95 import HausdorffDistance95
from .iou import IoU
from .mean_iou import MeanIoU
from .precision import Precision
from .recall import Recall

METRIC_FACTORIES = {
    "dice": Dice,
    "cldice": CLDice,
    "focal_loss": FocalLoss,
    "f1_score": F1Score,
    "iou": IoU,
    "mean_iou": MeanIoU,
    "precision": Precision,
    "recall": Recall,
    "hausdorff_distance": HausdorffDistance,
    "hausdorff_distance_95": HausdorffDistance95,
    "betti0_jaccard_ratio_gpu": Betti0JaccardRatioGPU,
    "betti0_variation_index_gpu": Betti0VariationIndexGPU,
    "betti1_jaccard_ratio_gpu": Betti1JaccardRatioGPU,
    "betti1_variation_index_gpu": Betti1VariationIndexGPU,
    "average_centerline_distance": AverageSymetricCenterlineDistance,
}


def get_metric(metric_config: dict) -> BaseMetric:
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
    result = {"cpu": [], "gpu": [], "mtg": []}
    for t in ["cpu", "gpu", "mtg"]:
        for cfg in metrics_config.get(t, []):
            metric = get_metric(cfg)
            result[t].append(metric)
    return result
