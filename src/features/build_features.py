"""Orchestration de la création de features (entraînement et inférence).

Deux contextes différents utilisent les features :

    * **Entraînement** — on lit un fichier parquet, on apprend les
      paramètres (médianes de marché par exemple) et on sauvegarde la
      matrice de features finale pour la passer au modèle.
    * **Inférence (production)** — on reçoit un (ou plusieurs) clients à
      scorer en direct ; on doit appliquer **exactement les mêmes**
      transformations que pendant l'entraînement, en utilisant le builder
      déjà entraîné.

Ce module fournit une fonction pour chacun de ces deux scénarios,
ainsi que la liste centralisée des colonnes à exclure des features
(:data:`NON_FEATURE_COLS`).
"""

# ───────────────────────────────────────────────────────
# RÔLE DE CE FICHIER (en clair) :
# C'est le "chef d'orchestre" qui appelle l'usine à features
# (ActuarialFeatureBuilder) au bon moment et avec les bonnes
# données : pendant l'entraînement, ou pendant le scoring en
# production. Il centralise aussi la liste des colonnes que
# le modèle ne doit pas voir (identifiants, libellés bruts,
# cible) pour éviter les fuites de données.
# ───────────────────────────────────────────────────────

from pathlib import Path

import pandas as pd

from src.features.actuarial_features import ActuarialFeatureBuilder

# Colonnes qu'on retire systématiquement avant de présenter les données
# au modèle. Soit elles sont des identifiants techniques sans valeur
# prédictive, soit elles existent déjà sous forme encodée (ex. ``lob``
# vs ``age_segment_encoded``), soit elles contiennent la réponse qu'on
# cherche à prédire (``churn_label``) — auquel cas les inclure
# constituerait une fuite de données catastrophique.
NON_FEATURE_COLS = [
    "policy_id", "customer_id", "inception_date", "expiry_date",
    "lob", "region", "channel",
    "churn_label",
]


def build_training_features(
    data_path: Path,
    output_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Construit la matrice d'entraînement à partir d'un parquet de clients.

    Cette fonction lit un fichier parquet, sépare la cible (``churn_label``)
    du reste, applique le pipeline de feature engineering et — si demandé —
    sauvegarde le tout sur disque.

    Args:
        data_path: Chemin vers le parquet de clients (au format produit
            par ``src/data/download_opendata.py``).
        output_path: Chemin où sauvegarder la matrice de features finale
            (avec la cible). Si ``None``, rien n'est sauvegardé.

    Returns:
        Une paire ``(X, y)``. ``X`` est la matrice numérique de features
        prête pour le modèle, ``y`` est la série cible (0 = reste,
        1 = part).
    """
    df = pd.read_parquet(data_path)

    y = df["churn_label"]
    X = df.drop(columns=["churn_label"])

    builder = ActuarialFeatureBuilder()
    X_features = builder.fit_transform(X)

    feature_cols = [c for c in X_features.columns if c not in NON_FEATURE_COLS]
    X_final = X_features[feature_cols]

    if output_path is not None:
        out = X_final.copy()
        out["churn_label"] = y.values
        out.to_parquet(output_path, index=False)

    return X_final, y


def build_inference_features(
    df: pd.DataFrame,
    builder: ActuarialFeatureBuilder,
) -> pd.DataFrame:
    """Construit la matrice de features pour l'inférence en production.

    À la différence de :func:`build_training_features`, on n'apprend pas
    de nouveaux paramètres ici : on réutilise un builder qui a déjà été
    entraîné. C'est crucial pour que les features produites en production
    soient cohérentes avec celles vues à l'entraînement.

    Args:
        df: DataFrame des clients à scorer (peut contenir une seule ligne).
        builder: Builder déjà entraîné (``fit`` déjà appelé). En général
            chargé depuis le fichier ``models/feature_builder.pkl`` produit
            par le script d'entraînement.

    Returns:
        DataFrame numérique de features, prêt à être passé au modèle.
    """
    df_features = builder.transform(df)
    feature_cols = [c for c in df_features.columns if c not in NON_FEATURE_COLS]
    return df_features[feature_cols]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Construit la matrice de features depuis un parquet brut."
    )
    parser.add_argument(
        "--input",
        default="data/processed/insurance_churn_train.parquet",
        help="Parquet d'entrée (clients au format standard)",
    )
    parser.add_argument(
        "--output",
        default="data/features.parquet",
        help="Parquet de sortie avec features + cible",
    )
    args = parser.parse_args()

    X, y = build_training_features(Path(args.input), Path(args.output))
    print(f"Matrice de features : {X.shape[0]} lignes, {X.shape[1]} colonnes")
    print(f"Taux de churn       : {y.mean():.1%}")
