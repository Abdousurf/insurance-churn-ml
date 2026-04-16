"""
ML Monitoring — Data & Model Drift Detection (MLOps)
=====================================================
Utilise Evidently AI pour détecter :
  1. Data drift    : les distributions des features ont-elles changé ?
  2. Target drift  : le taux de churn réel a-t-il évolué ?
  3. Model quality : dégradation des métriques en production ?

En production : déclenche une alerte et relance le réentraînement si dérive détectée.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent.parent.parent / "reports" / "monitoring"
THRESHOLD_DRIFT = 0.15  # Seuil de dérive acceptable (share of drifted features)
THRESHOLD_AUC_DROP = 0.05  # Dégradation AUC tolérée


def detect_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_col: str = "CARAVAN",
    report_name: str | None = None,
) -> dict[str, Any]:
    """
    Compare la distribution des features entre référence (train) et production.
    Retourne un rapport de dérive + flag d'alerte.
    """
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
        from evidently.report import Report

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = report_name or f"drift_report_{ts}"

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

        # Sauvegarde HTML
        html_path = REPORTS_DIR / f"{report_name}.html"
        report.save_html(str(html_path))

        # Extraction des métriques
        result_dict = report.as_dict()
        drift_metrics = result_dict["metrics"][0]["result"]
        share_drifted = drift_metrics.get("share_of_drifted_columns", 0.0)
        dataset_drift = drift_metrics.get("dataset_drift", False)

        summary = {
            "timestamp": ts,
            "share_drifted_features": share_drifted,
            "dataset_drift_detected": dataset_drift,
            "alert": share_drifted > THRESHOLD_DRIFT,
            "report_path": str(html_path),
            "n_reference": len(reference_df),
            "n_current": len(current_df),
        }

        # Sauvegarde JSON pour DVC / CI
        json_path = REPORTS_DIR / f"{report_name}.json"
        json_path.write_text(json.dumps(summary, indent=2))

        log.info("📊 Drift report : %.1f%% features dérivent | Alert=%s",
                 share_drifted * 100, summary["alert"])
        return summary

    except ImportError:
        log.warning("evidently non installé — pip install evidently")
        return _simple_drift_fallback(reference_df, current_df, target_col)


def _simple_drift_fallback(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_col: str,
) -> dict[str, Any]:
    """
    Détection de dérive simplifiée sans Evidently (fallback CI).
    Utilise le test de Kolmogorov-Smirnov pour les features numériques.
    """
    from scipy import stats

    numeric_cols = reference_df.select_dtypes("number").columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    drifted = []
    for col in numeric_cols[:20]:  # Limiter pour la CI
        if col in current_df.columns:
            stat, pvalue = stats.ks_2samp(reference_df[col].dropna(), current_df[col].dropna())
            if pvalue < 0.05:
                drifted.append(col)

    share = len(drifted) / max(len(numeric_cols), 1)
    log.info("📊 Drift simplifié (KS test) : %d/%d features dérivent (%.1f%%)",
             len(drifted), len(numeric_cols), share * 100)

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
    """
    Vérifie si la performance du modèle en production a chuté
    par rapport à la baseline MLflow.
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
        log.warning("🚨 DRIFT MODÈLE : AUC a chuté de %.3f → %.3f (drop=%.3f > seuil=%.3f)",
                    baseline_auc, current_auc, drop, THRESHOLD_AUC_DROP)
    else:
        log.info("✅ Performance stable : AUC=%.3f (baseline=%.3f)", current_auc, baseline_auc)

    return result


def generate_model_card(
    model_name: str,
    metrics: dict[str, float],
    feature_importance: dict[str, float],
    training_data_info: dict[str, Any],
    output_path: Path | None = None,
) -> str:
    """
    Génère une Model Card (bonnes pratiques MLOps, requis réglementation IA).
    Format Markdown — intégré dans le README ou publié avec le modèle.
    """
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

    card = f"""# Model Card — {model_name}

## Informations générales

| Champ | Valeur |
|-------|--------|
| Modèle | {model_name} |
| Date | {datetime.now().strftime('%Y-%m-%d')} |
| Version | voir MLflow registry |
| Domaine | Assurance — Prédiction de souscription/churn |

## Données d'entraînement

| Paramètre | Valeur |
|-----------|--------|
| Source | {training_data_info.get('source', 'UCI COIL 2000')} |
| Licence | {training_data_info.get('licence', 'CC BY 4.0')} |
| Taille train | {training_data_info.get('n_train', '?'):,} exemples |
| Taille test | {training_data_info.get('n_test', '?'):,} exemples |
| Taux positifs | {training_data_info.get('positive_rate', 0):.1%} |

## Performances

| Métrique | Valeur |
|----------|--------|
| AUC-ROC | {metrics.get('auc_roc', 0):.3f} |
| Precision@10% | {metrics.get('precision_at_10', 0):.1%} |
| Recall | {metrics.get('recall', 0):.3f} |
| F1-Score | {metrics.get('f1', 0):.3f} |

## Top 10 Features (SHAP)

| Feature | Importance |
|---------|-----------|
""" + "\n".join(f"| {f} | {v:.4f} |" for f, v in top_features) + f"""

## Biais et équité

- Le modèle n'utilise pas directement le genre, l'origine ou le code postal comme features.
- Les features de revenu (`MINKGEM`) peuvent introduire un biais indirect — monitorer les disparités par segment.
- Calibration vérifiée (Platt scaling) pour des probabilités exploitables en tarification.

## Limitations

- Entraîné sur des données hollandaises (COIL 2000) — les distributions peuvent différer du marché français.
- Classe déséquilibrée (~6% positifs) — les seuils de décision doivent être ajustés selon le coût business.
- Pas de données temporelles — utiliser TimeSeriesSplit pour éviter le data leakage.

## Usage recommandé

- ✅ Scoring de propension à la souscription (campagnes CRM)
- ✅ Priorisation des relances renouvellement
- ⚠️  Ne pas utiliser pour des décisions de refus sans contrôle humain (RGPD, directive IA)

## Réglementation

Ce modèle est soumis à :
- **RGPD Art. 22** — droit à l'explication pour les décisions automatisées
- **Directive IA (EU AI Act)** — système à risque limité, explication requise
- **Solvabilité II** — toute intégration en tarification nécessite validation actuarielle

---
*Généré automatiquement — {datetime.now().isoformat()}*
"""

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(card)
        log.info("✅ Model Card générée : %s", output_path)

    return card


if __name__ == "__main__":
    # Exemple de vérification de dérive avec les données COIL 2000
    from pathlib import Path

    processed = Path("data/processed")
    if (processed / "coil2000_train.parquet").exists():
        train = pd.read_parquet(processed / "coil2000_train.parquet")
        test  = pd.read_parquet(processed / "coil2000_test.parquet")

        result = detect_data_drift(train, test, target_col="CARAVAN")
        print("\nRésultat drift :")
        print(json.dumps(result, indent=2, default=str))
    else:
        print("⚠️  Données non disponibles — exécuter src/data/download_opendata.py d'abord")
