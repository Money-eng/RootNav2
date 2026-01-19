# Metrics/base.py

import abc


class BaseMetric(abc.ABC):
    """
    Interface minimale pour toutes les métriques :
    - chaque métrique doit définir un attribut de classe `type` valant "cpu" ou "gpu"
    - le constructeur __init__ peut prendre des arguments spécifiques
    - la méthode __call__(self, prediction: torch.Tensor, mask: torch.Tensor) -> float
      doit renvoyer un scalaire (float) correspondant à la valeur de la métrique sur ce batch.
    """
    type: str  # doit être "cpu" ou "gpu"

    @abc.abstractmethod
    def is_better(self, old_score: float, new_score: float) -> bool:
        """
        Compare deux scores de type float.
        Retourne True si `new_score` est meilleur que `old_score`.
        """
        pass

    @abc.abstractmethod
    def __call__(self, prediction, mask) -> float:
        """
        Calcule la métrique pour un batch de prédictions/tags.
        `prediction` et `mask` sont des torch.Tensor, même pour les métriques CPU :
        la conversion en numpy s’effectue à l’intérieur de la classe si besoin.
        Retourne un float.
        """
        pass
