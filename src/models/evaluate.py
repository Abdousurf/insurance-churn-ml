"""Évaluation du modèle : métriques techniques et métiers + visualisations.

Une fois le modèle entraîné, deux questions importent :

    * **Est-il juste, statistiquement parlant ?** On le vérifie avec des
      indicateurs classiques (AUC-ROC, average precision, score de Brier).
    * **Est-il utile pour le métier ?** On le vérifie avec des indicateurs
      "actionnables" : si on contacte les 15 % de clients les plus à
      risque, combien de vrais partants attrape-t-on ?

Ce module fournit les fonctions de calcul, les graphiques (lift, ROC,
calibration) et un *rapport complet* qui agrège l'ensemble. Il expose
aussi une CLI utilisée par le pipeline DVC pour publier un fichier de
métriques JSON et appliquer une "porte de qualité" : si l'AUC tombe
sous un seuil, le job CI échoue et le déploiement est bloqué.
"""

# ───────────────────────────────────────────────────────
# RÔLE DE CE FICHIER (en clair) :
# Il calcule "à quel point le modèle est bon" sous deux angles :
#   - une note technique (AUC-ROC, etc.) qui dit s'il classe
#     bien les clients,
#   - une note métier ("si on contacte les 15 % les plus à
#     risque, on attrape X % des vrais partants") qui parle
#     aux équipes commerciales.
# Il produit aussi des graphiques pour expliquer visuellement
# la performance.
# ───────────────────────────────────────────────────────

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # mode "headless" — n'ouvre pas de fenêtre graphique

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


def compute_business_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    top_k: float = 0.15,
) -> dict:
    """Calcule les métriques actionnables pour un seuil de ciblage donné.

    Question concrète : "si on contacte uniquement les ``top_k * 100`` %
    de clients que le modèle classe comme les plus à risque, quel est le
    rapport coût/efficacité ?"

    Args:
        y_true: Vraie issue pour chaque client (0 = resté, 1 = parti).
        y_prob: Probabilité de churn prédite par le modèle (entre 0 et 1).
        top_k: Fraction du portefeuille à cibler (par défaut 0.15 = 15 %).

    Returns:
        Dictionnaire contenant :
            ``precision`` (proportion de vrais partants dans la cible),
            ``recall``    (proportion de tous les partants attrapés),
            ``lift``      (combien de fois mieux que tirer au hasard),
            ``top_k``, ``churners_captured``, ``total_churners``.
    """
    n = len(y_true)
    k = int(n * top_k)

    sorted_idx = np.argsort(y_prob)[::-1]
    y_sorted = np.asarray(y_true)[sorted_idx]

    churners_in_top_k = y_sorted[:k].sum()
    total_churners = np.asarray(y_true).sum()

    precision_at_k = churners_in_top_k / k if k > 0 else 0.0
    recall_at_k = churners_in_top_k / total_churners if total_churners > 0 else 0.0
    base_rate = total_churners / n if n > 0 else 0.0
    lift = precision_at_k / base_rate if base_rate > 0 else 0.0

    return {
        "precision": round(precision_at_k, 4),
        "recall": round(recall_at_k, 4),
        "lift": round(lift, 2),
        "top_k": top_k,
        "churners_captured": int(churners_in_top_k),
        "total_churners": int(total_churners),
    }


def plot_lift_curve(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
    """Dessine la courbe de lift (gains cumulés vs ciblage aléatoire).

    Lecture : on trie les clients du plus à risque au moins à risque selon
    le modèle, puis on regarde, en avançant dans la liste, la part de
    vrais partants déjà rencontrés. Un bon modèle "remonte vite" la
    diagonale d'un tirage aléatoire.

    Args:
        y_true: Vraie issue pour chaque client (0/1).
        y_prob: Probabilité de churn prédite (0 à 1).

    Returns:
        La figure matplotlib avec la courbe modèle et la diagonale
        "tirage aléatoire".
    """
    n = len(y_true)
    sorted_idx = np.argsort(y_prob)[::-1]
    y_sorted = np.asarray(y_true)[sorted_idx]

    percentiles = np.arange(1, n + 1) / n
    cumulative_churners = np.cumsum(y_sorted)
    total_churners = y_sorted.sum()
    cumulative_recall = cumulative_churners / total_churners

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(percentiles * 100, cumulative_recall * 100, label="Modèle", linewidth=2)
    ax.plot([0, 100], [0, 100], "--", color="gray", label="Tirage aléatoire")
    ax.set_xlabel("% de la population ciblée")
    ax.set_ylabel("% de churners capturés")
    ax.set_title("Courbe de lift — gains cumulés")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> plt.Figure:
    """Dessine la courbe de calibration (probabilités prédites vs réelles).

    Question : quand le modèle dit "30 % de chance de partir", est-ce que
    30 % des clients de ce groupe partent vraiment ? Si la courbe colle à
    la diagonale, oui ; sinon, le modèle est mal calibré.

    Args:
        y_true: Vraie issue pour chaque client (0/1).
        y_prob: Probabilité de churn prédite (0 à 1).
        n_bins: Nombre de tranches de probabilité à comparer.

    Returns:
        La figure matplotlib avec la courbe et la diagonale parfaite.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Modèle")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Calibration parfaite")
    ax.set_xlabel("Probabilité prédite (moyenne)")
    ax.set_ylabel("Fraction de positifs réels")
    ax.set_title("Courbe de calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
    """Dessine la courbe ROC (taux de vrais positifs vs faux positifs).

    L'aire sous cette courbe (AUC) résume la capacité du modèle à
    distinguer les partants des non-partants : 1.0 = parfait, 0.5 =
    aléatoire. Au-dessus de 0.80, on parle d'un bon modèle pour ce type
    de problème déséquilibré.

    Args:
        y_true: Vraie issue pour chaque client (0/1).
        y_prob: Probabilité de churn prédite (0 à 1).

    Returns:
        La figure matplotlib avec la courbe et l'AUC en légende.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title("Courbe ROC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def full_evaluation_report(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Génère un rapport agrégé avec toutes les métriques techniques et métiers.

    Args:
        y_true: Vraie issue pour chaque client (0/1).
        y_prob: Probabilité de churn prédite (0 à 1).

    Returns:
        Dictionnaire avec ``auc_roc``, ``avg_precision``, ``brier_score``
        et les métriques métier aux seuils de ciblage 10 % et 15 %.
    """
    return {
        "auc_roc": round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision": round(average_precision_score(y_true, y_prob), 4),
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
        "business_metrics_10pct": compute_business_metrics(y_true, y_prob, top_k=0.10),
        "business_metrics_15pct": compute_business_metrics(y_true, y_prob, top_k=0.15),
    }


def _cli_evaluate(args) -> int:
    """Évalue le modèle Production sur un parquet de test et écrit le rapport.

    Cette fonction sert d'entrée à l'étape ``evaluate`` du pipeline DVC.
    Elle charge le modèle Production depuis MLflow, lui présente le jeu
    de test transformé par le builder de features sauvegardé, puis écrit
    les métriques au format JSON pour DVC et la CI. Renvoie 1 (échec) si
    l'AUC tombe sous le seuil ``--min-auc`` afin de bloquer un mauvais
    déploiement.

    Args:
        args: Arguments parsés depuis la ligne de commande.

    Returns:
        ``0`` si l'AUC dépasse ``--min-auc``, ``1`` sinon.
    """
    import json
    from pathlib import Path

    import pandas as pd

    from src.features.build_features import NON_FEATURE_COLS
    from src.models.predict import load_feature_builder, load_production_model

    df = pd.read_parquet(args.data_path)
    y_true = df["churn_label"].to_numpy()

    model = load_production_model()
    builder = load_feature_builder()
    df_features = builder.transform(df.drop(columns=["churn_label"]))
    feature_cols = [c for c in df_features.columns if c not in NON_FEATURE_COLS]
    y_prob = model.predict_proba(df_features[feature_cols])[:, 1]

    report = full_evaluation_report(y_true, y_prob)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Rapport d'evaluation -> {output_path}")
    print(f"AUC-ROC : {report['auc_roc']:.4f} (seuil : {args.min_auc})")

    return 0 if report["auc_roc"] >= args.min_auc else 1


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Evalue le modele Production sur un parquet de test."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Parquet de test contenant la colonne `churn_label`",
    )
    parser.add_argument(
        "--min-auc",
        type=float,
        default=0.70,
        help="Seuil de qualite (AUC) en deca duquel la CI echoue (defaut : %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="reports/metrics/eval_metrics.json",
        help="Chemin du rapport JSON a ecrire (defaut : %(default)s)",
    )
    sys.exit(_cli_evaluate(parser.parse_args()))
