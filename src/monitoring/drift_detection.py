"""Surveillance du modèle en production : dérive des données et performance.

Une fois le modèle déployé, deux phénomènes peuvent dégrader sa qualité
sans qu'on s'en aperçoive :

    1. **Dérive des données (data drift)** — la distribution des nouveaux
       clients diffère de celle vue à l'entraînement (changement de
       portefeuille, nouvelle campagne marketing, évolution démographique).
       Le modèle, qui n'a jamais vu ce profil, peut alors prédire de
       manière fiable mais sur la mauvaise population.

    2. **Dérive de performance (model drift)** — la performance mesurée
       sur des données fraîches chute par rapport à l'entraînement, par
       exemple parce qu'un évènement extérieur (concurrent, crise) modifie
       le comportement des clients.

Ce module fournit les outils pour détecter ces deux types de dérive et
générer une **model card** (fiche d'identité du modèle, requise par la
réglementation IA européenne et utile pour la conformité actuarielle).

L'outil principal pour la dérive des données est Evidently AI ; en
fallback (par exemple en CI où Evidently n'est pas installé), on
retombe sur un test de Kolmogorov-Smirnov simple.
"""

# ───────────────────────────────────────────────────────
# RÔLE DE CE FICHIER (en clair) :
# Il agit comme une "alerte qualité" pour le modèle en
# production. Il compare les nouveaux clients à ceux vus
# pendant l'entraînement : s'ils ressemblent à autre chose
# (nouvelle distribution), il déclenche une alerte. Il
# vérifie aussi si la performance du modèle (AUC) reste
# stable dans le temps.
# ───────────────────────────────────────────────────────

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# Dossier où sont stockés les rapports de monitoring (HTML + JSON).
REPORTS_DIR = Path(__file__).parent.parent.parent / "reports" / "monitoring"

# Seuil au-delà duquel on considère qu'il y a dérive : si plus de 15 %
# des features ont changé de distribution, c'est suspect.
THRESHOLD_DRIFT = 0.15

# Chute d'AUC tolérée avant de déclencher une alerte de re-entraînement.
THRESHOLD_AUC_DROP = 0.05


def detect_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_col: str = "churn_label",
    report_name: str | None = None,
) -> dict[str, Any]:
    """Compare les distributions de features entre une référence et un courant.

    "Référence" = données d'entraînement. "Courant" = données fraîches
    arrivées en production. La fonction génère un rapport HTML et JSON
    dans ``reports/monitoring/`` et renvoie un résumé exploitable par
    DVC ou un système d'alertes.

    Args:
        reference_df: Données de référence (typiquement le jeu d'entraînement).
        current_df: Données actuelles à comparer à la référence.
        target_col: Nom de la colonne cible (utilisée par Evidently pour
            calculer la dérive de la cible elle-même).
        report_name: Nom de base du fichier de rapport. Si ``None``, un
            nom horodaté est généré.

    Returns:
        Dictionnaire contenant la part de features dérivantes, le drapeau
        d'alerte, le chemin du rapport HTML et les tailles d'échantillons.
    """
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
        from evidently.report import Report

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = report_name or f"drift_report_{timestamp}"

        column_mapping = ColumnMapping(target=target_col)

        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
        ])
        report.run(
            reference_data=reference_df,
            current_data=current_df,
            column_mapping=column_mapping,
        )

        # Sauvegarde du rapport visuel HTML pour les humains.
        html_path = REPORTS_DIR / f"{report_name}.html"
        report.save_html(str(html_path))

        # Extraction des métriques clés pour la CI / DVC.
        result_dict = report.as_dict()
        drift_metrics = result_dict["metrics"][0]["result"]
        share_drifted = drift_metrics.get("share_of_drifted_columns", 0.0)
        dataset_drift = drift_metrics.get("dataset_drift", False)

        summary = {
            "timestamp": timestamp,
            "share_drifted_features": share_drifted,
            "dataset_drift_detected": dataset_drift,
            "alert": share_drifted > THRESHOLD_DRIFT,
            "report_path": str(html_path),
            "n_reference": len(reference_df),
            "n_current": len(current_df),
        }

        # Sauvegarde JSON pour DVC / CI.
        json_path = REPORTS_DIR / f"{report_name}.json"
        json_path.write_text(json.dumps(summary, indent=2))

        log.info(
            "Rapport de derive : %.1f%% des features ont change | Alerte=%s",
            share_drifted * 100, summary["alert"],
        )
        return summary

    except ImportError:
        log.warning("Evidently non installe — pip install evidently")
        return _simple_drift_fallback(reference_df, current_df, target_col)


def _simple_drift_fallback(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_col: str,
) -> dict[str, Any]:
    """Détection de dérive minimaliste sans Evidently (fallback CI).

    On utilise le test statistique de Kolmogorov-Smirnov sur chaque
    feature numérique : il compare les distributions de la référence et
    du courant et renvoie une p-value. Une p-value < 0.05 indique que
    les deux distributions diffèrent significativement.

    Args:
        reference_df: Données de référence (entraînement).
        current_df: Données actuelles.
        target_col: Colonne cible à exclure du calcul.

    Returns:
        Dictionnaire avec la part de features dérivantes, la liste des
        colonnes dérivées et le drapeau d'alerte.
    """
    from scipy import stats

    numeric_cols = reference_df.select_dtypes("number").columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    drifted: list[str] = []
    for col in numeric_cols[:20]:  # on limite pour ne pas alourdir la CI
        if col in current_df.columns:
            _, p_value = stats.ks_2samp(reference_df[col].dropna(), current_df[col].dropna())
            if p_value < 0.05:
                drifted.append(col)

    share = len(drifted) / max(len(numeric_cols), 1)
    log.info(
        "Derive simplifiee (test KS) : %d/%d features ont change (%.1f%%)",
        len(drifted), len(numeric_cols), share * 100,
    )

    return {
        "method": "ks_test_fallback",
        "share_drifted_features": share,
        "drifted_columns": drifted,
        "alert": share > THRESHOLD_DRIFT,
        "dataset_drift_detected": share > THRESHOLD_DRIFT,
    }


def check_model_performance_drift(
    current_auc: float,
    baseline_auc: float,
    metrics_path: Path | None = None,
) -> dict[str, Any]:
    """Vérifie la dégradation éventuelle de l'AUC en production.

    On compare l'AUC mesurée sur les données fraîches à l'AUC de référence
    (typiquement celle obtenue à l'entraînement et stockée dans MLflow).
    Une chute supérieure à :data:`THRESHOLD_AUC_DROP` déclenche une
    recommandation de ré-entraînement.

    Args:
        current_auc: AUC mesurée récemment.
        baseline_auc: AUC obtenue à l'entraînement (référence).
        metrics_path: Chemin où sauvegarder le résultat en JSON
            (facultatif).

    Returns:
        Dictionnaire avec ``baseline_auc``, ``current_auc``, ``auc_drop``,
        ``alert`` et ``recommendation`` (``"retrain"`` ou ``"ok"``).
    """
    drop = baseline_auc - current_auc
    alert = drop > THRESHOLD_AUC_DROP

    result = {
        "baseline_auc": baseline_auc,
        "current_auc": current_auc,
        "auc_drop": drop,
        "alert": alert,
        "recommendation": "retrain" if alert else "ok",
    }

    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(result, indent=2))

    if alert:
        log.warning(
            "DERIVE MODELE : AUC chute de %.3f -> %.3f (drop=%.3f > seuil=%.3f)",
            baseline_auc, current_auc, drop, THRESHOLD_AUC_DROP,
        )
    else:
        log.info("Performance stable : AUC=%.3f (baseline=%.3f)", current_auc, baseline_auc)

    return result


def generate_model_card(
    model_name: str,
    metrics: dict[str, float],
    feature_importance: dict[str, float],
    training_data_info: dict[str, Any],
    output_path: Path | None = None,
) -> str:
    """Génère une "model card" (fiche d'identité du modèle) en Markdown.

    Une model card est un document standardisé qui présente le modèle,
    ses performances, ses limites, ses biais potentiels et son cadre
    réglementaire. Elle est exigée par le règlement européen sur l'IA
    et fait partie des bonnes pratiques MLOps.

    Args:
        model_name: Nom du modèle (ex. ``"insurance_churn_xgb"``).
        metrics: Dictionnaire des métriques d'évaluation.
        feature_importance: Importance SHAP par feature ; les 10
            premières sont listées dans la fiche.
        training_data_info: Métadonnées du dataset d'entraînement
            (source, licence, taille, taux de positifs).
        output_path: Si fourni, écrit la fiche en Markdown sur disque.

    Returns:
        Le contenu Markdown de la fiche.
    """
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

    card = f"""# Model Card — {model_name}

## Informations générales

| Champ | Valeur |
|-------|--------|
| Modèle | {model_name} |
| Date | {datetime.now().strftime('%Y-%m-%d')} |
| Version | voir registre MLflow |
| Domaine | Assurance — prédiction de souscription / churn |

## Données d'entraînement

| Paramètre | Valeur |
|-----------|--------|
| Source | {training_data_info.get('source', 'UCI COIL 2000')} |
| Licence | {training_data_info.get('licence', 'CC BY 4.0')} |
| Taille train | {training_data_info.get('n_train', '?'):,} exemples |
| Taille test | {training_data_info.get('n_test', '?'):,} exemples |
| Taux de positifs | {training_data_info.get('positive_rate', 0):.1%} |

## Performances

| Métrique | Valeur |
|----------|--------|
| AUC-ROC | {metrics.get('auc_roc', 0):.3f} |
| Precision @ 10 % | {metrics.get('precision_at_10', 0):.1%} |
| Rappel | {metrics.get('recall', 0):.3f} |
| F1-Score | {metrics.get('f1', 0):.3f} |

## Top 10 features (importance SHAP)

| Feature | Importance |
|---------|-----------|
""" + "\n".join(f"| {f} | {v:.4f} |" for f, v in top_features) + """

## Biais et équité

- Le modèle n'utilise pas directement le genre, l'origine ou le code
  postal comme features.
- Les features de revenu (``MINKGEM``) peuvent introduire un biais
  indirect — il faut suivre les écarts de prédiction par segment de
  revenu.
- La calibration a été vérifiée (Platt scaling) afin que les
  probabilités produites soient exploitables côté tarification.

## Limites connues

- Modèle entraîné sur des données hollandaises (COIL 2000) — les
  distributions peuvent différer du marché français.
- Classes déséquilibrées (~6 % de positifs) — ajuster le seuil de
  décision selon le coût business.
- Pas d'historique temporel propre : utiliser ``TimeSeriesSplit`` à
  l'évaluation pour éviter les fuites de données.

## Usage recommandé

- ✅ Scoring de propension à la souscription (campagnes CRM).
- ✅ Priorisation des relances de renouvellement.
- ⚠️  Ne pas utiliser pour des décisions de refus sans contrôle
  humain (RGPD, directive IA).

## Cadre réglementaire

Ce modèle est soumis à :

- **RGPD Art. 22** — droit à l'explication pour les décisions
  automatisées.
- **Règlement européen sur l'IA (AI Act)** — système à risque limité,
  explication requise.
- **Solvabilité II** — toute intégration en tarification nécessite une
  validation actuarielle.

---
*Fiche générée automatiquement — """ + datetime.now().isoformat() + "*\n"

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(card)
        log.info("Model card generee : %s", output_path)

    return card


if __name__ == "__main__":
    # Exécution de démo : compare les parquets train et test produits par
    # ``src/data/download_opendata.py`` et publie le rapport de dérive.
    processed = Path("data/processed")
    train_file = processed / "insurance_churn_train.parquet"
    test_file = processed / "insurance_churn_test.parquet"

    if train_file.exists() and test_file.exists():
        train = pd.read_parquet(train_file)
        test = pd.read_parquet(test_file)

        result = detect_data_drift(train, test, target_col="churn_label")
        print("\nResultat de la detection de derive :")
        print(json.dumps(result, indent=2, default=str))
    else:
        print("Donnees indisponibles — lancer src/data/download_opendata.py d'abord.")
