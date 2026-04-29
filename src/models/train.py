"""Entraînement du modèle de prédiction de churn.

Le script entraîne **deux modèles** et garde le meilleur :

    1. Un modèle simple (régression logistique) qui sert de **plancher**
       de qualité — il faut le battre clairement pour mériter d'être mis
       en production.
    2. Un modèle plus puissant (XGBoost) dont les réglages sont **ajustés
       automatiquement** par Optuna, qui essaie plusieurs dizaines de
       combinaisons et retient la meilleure.

Tous les résultats (métriques, hyperparamètres, courbes) sont enregistrés
dans MLflow afin qu'on puisse comparer les expériences au fil du temps.
Le modèle gagnant est stocké dans le **registre MLflow**, et le pipeline
de feature engineering qu'il a vu à l'entraînement est sauvegardé sur
disque pour que l'API puisse appliquer **exactement les mêmes
transformations** lors de l'inférence.
"""

# ───────────────────────────────────────────────────────
# RÔLE DE CE FICHIER (en clair) :
# Ce script lance l'entraînement de bout en bout :
#   1. Il lit le fichier de clients.
#   2. Il fabrique les variables explicatives (features).
#   3. Il entraîne un premier modèle simple comme repère.
#   4. Il entraîne un modèle plus performant et tente plein
#      de réglages pour trouver le meilleur.
#   5. Il enregistre tous les résultats dans MLflow.
#   6. Il sauvegarde le modèle gagnant et son pipeline de
#      features pour que l'API puisse les recharger ensuite.
# ───────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier

from src.features.actuarial_features import ActuarialFeatureBuilder
from src.features.build_features import NON_FEATURE_COLS
from src.models.evaluate import compute_business_metrics, plot_lift_curve

# On masque les avertissements de bibliothèque qui polluent la sortie.
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Configuration (pilotée par variables d'environnement) ───────────────────

# Adresse du serveur MLflow. Surcharger avec la variable d'environnement
# ``MLFLOW_TRACKING_URI`` si MLflow tourne ailleurs que sur localhost.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Parquet d'entraînement par défaut — produit par
# ``src/data/download_opendata.py``.
DEFAULT_DATA_PATH = "data/processed/insurance_churn_train.parquet"

# Emplacement où sauvegarder le pipeline de feature engineering pour que
# l'API puisse le recharger.
FEATURE_BUILDER_PATH = Path(os.getenv("FEATURE_BUILDER_PATH", "models/feature_builder.pkl"))

# Nom sous lequel le modèle est enregistré dans le registre MLflow.
REGISTERED_MODEL_NAME = "insurance_churn_xgb"


# ── Lecture des données ─────────────────────────────────────────────────────


def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Lit le parquet de clients et sépare features et cible.

    Args:
        data_path: Chemin vers le fichier parquet des clients
            (produit par ``src/data/download_opendata.py``).

    Returns:
        Une paire ``(features, cible)``. ``features`` est tout sauf la
        colonne ``churn_label`` ; ``cible`` est la série binaire
        "0 = est resté, 1 = est parti".
    """
    df = pd.read_parquet(data_path)
    labels = df["churn_label"]
    features = df.drop(columns=["churn_label"])
    return features, labels


# ── Modèles ─────────────────────────────────────────────────────────────────


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Entraîne une régression logistique comme plancher de qualité.

    Avoir un baseline garde l'équipe honnête : un modèle plus complexe
    n'a d'intérêt que s'il bat clairement ce baseline.

    Args:
        X_train: Matrice de features d'entraînement (uniquement numérique).
        y_train: Cible d'entraînement (0 = resté, 1 = parti).

    Returns:
        Le modèle de régression logistique entraîné.
    """
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model


def _xgb_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: TimeSeriesSplit,
) -> float:
    """Évalue une combinaison de réglages XGBoost pour Optuna.

    Cette fonction est appelée des dizaines de fois par Optuna : à chaque
    appel, elle reçoit une nouvelle proposition de réglages, entraîne un
    modèle avec ces réglages, et renvoie le score moyen obtenu en
    validation croisée. Optuna garde la combinaison qui maximise ce score.

    Args:
        trial: Interface Optuna pour échantillonner des valeurs de
            hyperparamètres dans des plages prédéfinies.
        X_train: Matrice de features d'entraînement.
        y_train: Cible d'entraînement.
        cv: Stratégie de découpage en folds (ici un découpage temporel
            pour ne pas "voir le futur").

    Returns:
        Le score AUC-ROC moyen obtenu en validation croisée.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 5),
        "random_state": 42,
        "eval_metric": "auc",
        "verbosity": 0,
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


def train_with_optuna(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
) -> tuple[XGBClassifier, float]:
    """Optimise les hyperparamètres XGBoost puis entraîne le modèle final.

    Optuna explore ``n_trials`` combinaisons de réglages et retient celle
    qui obtient le meilleur score en validation croisée. On entraîne
    ensuite un modèle frais avec cette combinaison gagnante, sur **toute**
    la base d'entraînement.

    Args:
        X_train: Matrice de features d'entraînement.
        y_train: Cible d'entraînement.
        n_trials: Nombre de combinaisons de réglages à tester. Plus c'est
            grand, plus on a de chances de trouver une bonne combinaison,
            mais l'entraînement est plus long.

    Returns:
        Une paire ``(modèle, meilleur_score)``. Le modèle est prêt à
        prédire ; ``meilleur_score`` est l'AUC obtenu en validation
        croisée par la combinaison gagnante.
    """
    cv = TimeSeriesSplit(n_splits=5)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _xgb_objective(trial, X_train, y_train, cv),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_params.update({"random_state": 42, "eval_metric": "auc", "verbosity": 0})
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    return model, float(study.best_value)


# ── Persistance du builder de features ──────────────────────────────────────


def _save_feature_builder(builder: ActuarialFeatureBuilder, path: Path) -> Path:
    """Sauvegarde le builder de features sur disque pour l'API.

    Le builder a appris à l'entraînement la prime médiane par branche.
    L'API a besoin de cette même information pour calculer le ratio
    "prime / médiane de marché" en inférence — sinon les features
    produites différeraient silencieusement de celles vues à
    l'entraînement, dégradant la qualité des prédictions.

    Args:
        builder: Un :class:`ActuarialFeatureBuilder` déjà entraîné
            (``fit`` ou ``fit_transform`` déjà appelé).
        path: Chemin où sauvegarder le fichier pickle.

    Returns:
        Le chemin du fichier écrit (utile pour les logs).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(builder, path)
    return path


# ── Pipeline principal ──────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    """Exécute le pipeline d'entraînement complet de bout en bout.

    Args:
        args: Arguments parsés depuis la ligne de commande
            (voir :func:`_build_parser`).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(args.experiment_name)

    print(f"Lecture des donnees depuis {args.data_path}...")
    X, y = load_data(Path(args.data_path))

    # On entraîne le builder de features et on garde l'instance pour la
    # sauvegarder plus tard.
    builder = ActuarialFeatureBuilder()
    X_features = builder.fit_transform(X)
    feature_cols = [c for c in X_features.columns if c not in NON_FEATURE_COLS]
    X_features = X_features[feature_cols]

    # Découpe chronologique : on entraîne sur les 80 % les plus anciens et
    # on teste sur les 20 % les plus récents. Un découpage aléatoire ferait
    # "voir le futur" au modèle et gonflerait artificiellement les scores.
    split_idx = int(len(X_features) * 0.80)
    X_train, X_test = X_features.iloc[:split_idx], X_features.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"Train : {len(X_train):,} | Test : {len(X_test):,} | Taux churn : {y.mean():.1%}")

    # ── Run 1 : baseline ─────────────────────────────────────────────────
    with mlflow.start_run(run_name="logistic_baseline"):
        baseline = train_baseline(X_train, y_train)
        y_prob_base = baseline.predict_proba(X_test)[:, 1]
        auc_base = roc_auc_score(y_test, y_prob_base)
        ap_base = average_precision_score(y_test, y_prob_base)
        brier_base = brier_score_loss(y_test, y_prob_base)

        mlflow.log_params({"model_type": "logistic_regression", "class_weight": "balanced"})
        mlflow.log_metrics({
            "auc_roc": auc_base,
            "avg_precision": ap_base,
            "brier_score": brier_base,
        })
        mlflow.sklearn.log_model(baseline, name="model")
        print(f"AUC baseline : {auc_base:.4f}")

    # ── Run 2 : XGBoost optimisé par Optuna ──────────────────────────────
    with mlflow.start_run(run_name="xgboost_optuna"):
        print(f"Tuning XGBoost avec {args.n_trials} essais Optuna...")
        xgb_model, best_cv_auc = train_with_optuna(X_train, y_train, args.n_trials)

        # Re-calibration des probabilités : on s'assure qu'une prédiction
        # "30 % de chance de partir" correspond bien à ~30 % de départs
        # réels dans la population concernée. ``FrozenEstimator`` indique
        # au calibrateur de ne PAS ré-entraîner XGBoost.
        calibrated = CalibratedClassifierCV(FrozenEstimator(xgb_model), method="sigmoid")
        calibrated.fit(X_train, y_train)

        y_prob_xgb = calibrated.predict_proba(X_test)[:, 1]
        auc_xgb = roc_auc_score(y_test, y_prob_xgb)
        ap_xgb = average_precision_score(y_test, y_prob_xgb)
        brier_xgb = brier_score_loss(y_test, y_prob_xgb)

        # Métriques métier : si on contacte les 15 % de clients les plus à
        # risque, combien de vrais "partants" attrape-t-on ?
        business_metrics = compute_business_metrics(y_test, y_prob_xgb, top_k=0.15)

        mlflow.log_params({
            "model_type": "xgboost_calibrated",
            "n_trials": args.n_trials,
            **xgb_model.get_params(),
        })
        mlflow.log_metrics({
            "auc_roc": auc_xgb,
            "cv_auc_best": best_cv_auc,
            "avg_precision": ap_xgb,
            "brier_score": brier_xgb,
            "lift_at_15pct": business_metrics["lift"],
            "recall_at_15pct": business_metrics["recall"],
            "precision_at_15pct": business_metrics["precision"],
        })

        # SHAP : graphe expliquant quelles features pèsent le plus dans
        # les décisions du modèle. La fonction de SHAP écrit sur la
        # figure matplotlib courante au lieu de la renvoyer, donc on la
        # capture via ``plt.gcf()``.
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False)
        mlflow.log_figure(plt.gcf(), "shap_summary.png")
        plt.close("all")

        # Courbe de lift : "si on contacte les X % les plus à risque,
        # quelle proportion de vrais partants attrape-t-on ?"
        mlflow.log_figure(plot_lift_curve(y_test, y_prob_xgb), "lift_curve.png")

        # On sauvegarde le builder de features à côté du modèle pour que
        # l'API utilise EXACTEMENT les mêmes références marché.
        builder_file = _save_feature_builder(builder, FEATURE_BUILDER_PATH)
        mlflow.log_artifact(str(builder_file))

        # Enregistrement du modèle calibré dans le registre MLflow ; l'API
        # le récupèrera ensuite par son nom logique.
        mlflow.sklearn.log_model(
            calibrated,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print("\n── Resultats ──────────────────────────────────")
        print(f"  AUC-ROC XGBoost     : {auc_xgb:.4f}")
        print(f"  Lift top 15 %       : {business_metrics['lift']:.2f}x")
        print(f"  Rappel top 15 %     : {business_metrics['recall']:.1%}")
        print(f"  AUC-ROC baseline    : {auc_base:.4f}")
        print(f"  Amelioration        : +{(auc_xgb - auc_base):.4f}")
        print(f"  Builder de features : {builder_file}")


def _build_parser() -> argparse.ArgumentParser:
    """Construit le parseur des arguments de ligne de commande.

    Returns:
        Le parseur configuré, prêt à appeler ``.parse_args()``.
    """
    parser = argparse.ArgumentParser(description="Entraine le modele de churn assurance.")
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Parquet d'entree avec les fiches clients (defaut : %(default)s)",
    )
    parser.add_argument(
        "--experiment-name",
        default="churn_v1",
        help="Nom de l'experience MLflow (defaut : %(default)s)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Nombre d'essais d'optimisation Optuna (defaut : %(default)s)",
    )
    return parser


if __name__ == "__main__":
    main(_build_parser().parse_args())
