# Metrics/base.py

import abc


class BaseMetric(abc.ABC):
    type: str  # must be "gpu" or "cpu"

    @abc.abstractmethod
    def is_better(self, old_score: float, new_score: float) -> bool:
        pass

    @abc.abstractmethod
    def __call__(self, prediction, mask) -> float:
        pass
