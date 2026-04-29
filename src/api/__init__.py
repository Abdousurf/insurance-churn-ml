"""Sous-package "api" : service web FastAPI qui sert les prédictions.

Le service expose quatre endpoints HTTP :

    * ``GET  /health``         — état du service
    * ``GET  /model-info``     — métadonnées du modèle chargé
    * ``POST /predict``        — score d'un client
    * ``POST /predict/batch``  — score d'une liste de clients

Symboles publics ré-exportés ici :

    * :data:`app`                     — l'application FastAPI prête à servir
    * :class:`PolicyFeatures`         — schéma d'un client en entrée
    * :class:`ChurnPrediction`        — schéma d'une prédiction en sortie
    * :class:`BatchRequest`, :class:`BatchResponse` — schémas du mode batch
    * :func:`classify_risk`, :func:`estimate_clv`, :func:`features_to_dataframe`
    * :data:`RETENTION_ACTIONS`, :data:`RISK_TIERS`
"""

from src.api.main import app
from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    ChurnPrediction,
    PolicyFeatures,
)
from src.api.utils import (
    RETENTION_ACTIONS,
    RISK_TIERS,
    classify_risk,
    estimate_clv,
    features_to_dataframe,
)

__all__ = [
    "BatchRequest",
    "BatchResponse",
    "ChurnPrediction",
    "PolicyFeatures",
    "RETENTION_ACTIONS",
    "RISK_TIERS",
    "app",
    "classify_risk",
    "estimate_clv",
    "features_to_dataframe",
]
