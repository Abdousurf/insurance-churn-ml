"""Sous-package "monitoring" : surveillance du modèle en production.

Une fois le modèle déployé, deux risques principaux apparaissent :

    1. **Dérive des données** — la distribution des nouveaux clients
       diffère de celle vue à l'entraînement (changement de portefeuille,
       nouvelle campagne marketing…). Le modèle, qui n'a jamais vu ce
       genre de profils, peut alors prédire faux sans qu'on le sache.

    2. **Dérive de performance** — l'AUC mesurée sur des données
       fraîches chute par rapport au niveau d'entraînement. C'est un
       signe que le modèle n'est plus adapté et qu'il faut le ré-entraîner.

Ce sous-package fournit les outils pour détecter ces deux types de
dérive et générer une "model card" (fiche d'identité du modèle, requise
par la réglementation IA européenne).

Symboles publics ré-exportés ici :

    * :func:`detect_data_drift`              — détecte la dérive des features
    * :func:`check_model_performance_drift`  — détecte la chute d'AUC
    * :func:`generate_model_card`            — génère la fiche d'identité
"""

from src.monitoring.drift_detection import (
    check_model_performance_drift,
    detect_data_drift,
    generate_model_card,
)

__all__ = [
    "check_model_performance_drift",
    "detect_data_drift",
    "generate_model_card",
]
