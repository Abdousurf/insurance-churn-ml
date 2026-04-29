"""Sous-package "models" : entraînement, évaluation et inférence du modèle.

Trois rôles principaux :

    * **Entraînement** (``train.py``) — apprend un modèle sur les données
      historiques et le sauvegarde dans MLflow.
    * **Évaluation** (``evaluate.py``) — mesure la qualité du modèle avec
      des indicateurs techniques (AUC, Brier) et métiers (lift @ top 15 %).
    * **Inférence** (``predict.py``) — utilise un modèle déjà entraîné pour
      scorer de nouveaux clients.

Symboles publics ré-exportés ici :

    * :func:`load_data`, :func:`train_baseline`, :func:`train_with_optuna`
    * :func:`compute_business_metrics`, :func:`full_evaluation_report`
    * :func:`plot_lift_curve`, :func:`plot_calibration`, :func:`plot_roc_curve`
    * :func:`load_production_model`, :func:`load_feature_builder`
    * :func:`predict_dataframe`, :func:`batch_score`
"""

from src.models.evaluate import (
    compute_business_metrics,
    full_evaluation_report,
    plot_calibration,
    plot_lift_curve,
    plot_roc_curve,
)
from src.models.predict import (
    batch_score,
    load_feature_builder,
    load_production_model,
    predict_dataframe,
)
from src.models.train import (
    load_data,
    train_baseline,
    train_with_optuna,
)

__all__ = [
    "batch_score",
    "compute_business_metrics",
    "full_evaluation_report",
    "load_data",
    "load_feature_builder",
    "load_production_model",
    "plot_calibration",
    "plot_lift_curve",
    "plot_roc_curve",
    "predict_dataframe",
    "train_baseline",
    "train_with_optuna",
]
