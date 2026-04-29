"""Package racine du projet de prédiction de churn assurance.

Ce dossier regroupe tout le code source du projet, organisé en
sous-packages spécialisés :

    * ``data``        — téléchargement et préparation des données brutes
    * ``features``    — création des variables explicatives "métier"
    * ``models``      — entraînement, évaluation et inférence du modèle
    * ``api``         — service web FastAPI qui expose les prédictions
    * ``monitoring``  — surveillance de la qualité du modèle en production

Chaque sous-package ré-expose ses symboles publics dans son ``__init__.py``
afin de simplifier les imports : on peut écrire ``from src.features import
ActuarialFeatureBuilder`` plutôt que de chercher le bon module interne.
"""
