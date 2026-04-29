"""Inférence en lot : score un fichier complet de clients avec le modèle.

Ce module sert principalement aux usages "batch" (campagnes CRM
nocturnes, exports planifiés) où l'on veut scorer plusieurs milliers de
clients d'un coup et écrire les résultats dans un fichier parquet.

Pour l'inférence en temps réel, voir le service FastAPI dans
:mod:`src.api`.

Le modèle est chargé depuis le **registre MLflow** (étape ``Production``)
et le builder de features est chargé depuis le disque (fichier
sauvegardé pendant l'entraînement). Cela garantit que les
transformations appliquées en inférence sont **strictement identiques**
à celles vues à l'entraînement.
"""

# ───────────────────────────────────────────────────────
# RÔLE DE CE FICHIER (en clair) :
# C'est l'outil "scorez-moi tout le portefeuille". On lui
# donne un fichier de clients, il sort un fichier avec, pour
# chaque client : sa probabilité de partir (0 à 100 %) et un
# niveau de risque (low / medium / high / critical).
# ───────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from src.features.actuarial_features import ActuarialFeatureBuilder
from src.features.build_features import NON_FEATURE_COLS

# Configuration via variables d'environnement pour qu'un même code tourne
# en local et en CI sans modification.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "insurance_churn_xgb")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
FEATURE_BUILDER_PATH = Path(os.getenv("FEATURE_BUILDER_PATH", "models/feature_builder.pkl"))


def load_production_model():
    """Charge le modèle Production depuis le registre MLflow.

    Returns:
        Le modèle entraîné, prêt à appeler ``predict_proba``.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")


def load_feature_builder() -> ActuarialFeatureBuilder:
    """Charge le builder de features sauvegardé pendant l'entraînement.

    Si le fichier n'existe pas (ex. on lance l'inférence avant d'avoir
    entraîné le modèle), on retombe sur un builder neuf, non entraîné.
    L'inférence reste possible mais la feature "prime vs marché" sera
    dégradée — un signal clair qu'il manque une étape.

    Returns:
        Un :class:`ActuarialFeatureBuilder`, idéalement déjà entraîné.
    """
    if FEATURE_BUILDER_PATH.exists():
        return joblib.load(FEATURE_BUILDER_PATH)
    return ActuarialFeatureBuilder()


def predict_dataframe(
    df: pd.DataFrame,
    model,
    builder: ActuarialFeatureBuilder,
) -> pd.DataFrame:
    """Score un tableau de clients et leur attribue un niveau de risque.

    Args:
        df: Tableau de clients au format standard du projet (colonnes
            ``policy_id``, ``lob``, ``annual_premium``, …).
        model: Modèle entraîné (typiquement chargé via
            :func:`load_production_model`).
        builder: Builder de features cohérent avec l'entraînement.

    Returns:
        DataFrame de résultats avec trois colonnes : ``policy_id`` (si
        présent dans l'entrée), ``churn_probability`` et ``risk_tier``.
    """
    df_features = builder.transform(df)
    feature_cols = [c for c in df_features.columns if c not in NON_FEATURE_COLS]
    X = df_features[feature_cols]

    probabilities = model.predict_proba(X)[:, 1]

    # On garde l'identifiant du client si on l'a, sinon on conserve juste
    # l'index d'origine pour pouvoir rapprocher les lignes.
    result = (
        df[["policy_id"]].copy()
        if "policy_id" in df.columns
        else pd.DataFrame(index=df.index)
    )
    result["churn_probability"] = probabilities
    result["risk_tier"] = pd.cut(
        probabilities,
        bins=[0, 0.20, 0.45, 0.70, 1.0],
        labels=["low", "medium", "high", "critical"],
        include_lowest=True,
    )
    return result


def batch_score(
    input_path: Path,
    output_path: Path,
    model=None,
    builder: ActuarialFeatureBuilder | None = None,
) -> pd.DataFrame:
    """Lit un parquet de clients, les score tous, écrit le résultat sur disque.

    Args:
        input_path: Chemin du fichier parquet contenant les clients à scorer.
        output_path: Chemin où sauvegarder les prédictions.
        model: Modèle déjà chargé. S'il est ``None``, on charge le modèle
            Production via :func:`load_production_model`.
        builder: Builder déjà chargé. S'il est ``None``, on charge celui
            du disque via :func:`load_feature_builder`.

    Returns:
        Le même DataFrame de résultats que celui écrit sur disque.
    """
    if model is None:
        model = load_production_model()
    if builder is None:
        builder = load_feature_builder()

    df = pd.read_parquet(input_path)
    results = predict_dataframe(df, model, builder)
    results.to_parquet(output_path, index=False)
    print(f"{len(results)} clients scores -> {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scoring batch des assures avec le modele de churn."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Parquet d'entree contenant les clients a scorer",
    )
    parser.add_argument(
        "--output",
        default="data/scores.parquet",
        help="Parquet de sortie pour les predictions (defaut : %(default)s)",
    )
    args = parser.parse_args()
    batch_score(Path(args.input), Path(args.output))
