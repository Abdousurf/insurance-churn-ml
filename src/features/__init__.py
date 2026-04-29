"""Sous-package "features" : création des variables que le modèle apprend.

Le modèle ne consomme pas directement les colonnes brutes (prime, âge,
ancienneté…). On les transforme d'abord en indicateurs "métier" plus
parlants : "ce client paie-t-il au-dessus du marché ?", "est-il un
nouveau client ?", "a-t-il plusieurs contrats ?". C'est le rôle de ce
sous-package.

Symboles publics ré-exportés ici :

    * :class:`ActuarialFeatureBuilder` — pipeline scikit-learn de feature engineering
    * :func:`build_feature_matrix`     — raccourci pour produire une matrice prête à l'usage
    * :func:`build_training_features`  — version "fichier parquet" pour l'entraînement
    * :func:`build_inference_features` — version "live" pour le scoring en production
    * :data:`NON_FEATURE_COLS`         — colonnes à exclure du modèle (ID, libellés, cible)
    * :data:`FEATURE_DESCRIPTIONS`     — dictionnaire descriptif des features (pour la doc)
"""

from src.features.actuarial_features import (
    FEATURE_DESCRIPTIONS,
    ActuarialFeatureBuilder,
    build_feature_matrix,
)
from src.features.build_features import (
    NON_FEATURE_COLS,
    build_inference_features,
    build_training_features,
)

__all__ = [
    "ActuarialFeatureBuilder",
    "FEATURE_DESCRIPTIONS",
    "NON_FEATURE_COLS",
    "build_feature_matrix",
    "build_inference_features",
    "build_training_features",
]
