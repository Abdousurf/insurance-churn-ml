# Prédiction de churn assurance & optimisation des primes

> **Pipeline ML de bout en bout** — prédire les assurés sur le point de partir et calibrer les actions de rétention à partir de variables actuarielles.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.9-red)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-pret-blue)](https://docker.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![CI](https://github.com/Abdousurf/insurance-churn-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdousurf/insurance-churn-ml/actions)
[![DVC](https://img.shields.io/badge/DVC-versionne-945dd6)](https://dvc.org)
[![Open Data](https://img.shields.io/badge/Open%20Data-UCI%20COIL%202000-green)](https://archive.ics.uci.edu/dataset/125/insurance+company+benchmark+coil+2000)

## Source de données ouverte

**Dataset : UCI Insurance Company Benchmark — COIL 2000**

| Attribut | Détail |
|----------|--------|
| Source | UCI Machine Learning Repository |
| URL | [archive.ics.uci.edu/dataset/125](https://archive.ics.uci.edu/dataset/125/insurance+company+benchmark+coil+2000) |
| Licence | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Volume | 9 068 clients · 86 variables · cible binaire (~6 % de positifs) |
| Variables | Socio-démographie + types de contrats + historique sinistres |
| Pertinence | Classification binaire déséquilibrée — réaliste pour l'assurance |

Le script `src/data/download_opendata.py` télécharge le dataset puis le **traduit** vers le schéma "fiche client" standard du projet (prime annuelle, ancienneté, branche, canal d'acquisition, etc.). Ainsi, toutes les couches en aval — feature engineering, entraînement, API, monitoring — voient la **même structure** de données.

## Problème métier

En assurance, **un client qui part = prime perdue + coût d'acquisition** pour le remplacer. Réduire le churn de 5 % sur un portefeuille de 50 M€ représente environ 3 M€ d'impact direct. Ce projet construit un système de prédiction de churn prêt pour la production avec :

1. Un classifieur binaire pour identifier les assurés à risque (AUC > 0,82)
2. Un score de risque qui pilote des offres de rétention personnalisées
3. Une API REST pour l'intégration en temps réel avec le CRM

## Architecture du pipeline ML

```
Données brutes
    │
    ├─ Feature engineering ──► Variables actuarielles (inspirées GLM)
    │                          Variables comportementales (sinistralité)
    │                          Variables contractuelles (ancienneté, prime)
    │
    ├─ Entraînement ─────────► XGBoost + LightGBM + régression logistique
    │                          Tuning des hyperparamètres (Optuna)
    │                          Validation croisée (TimeSeriesSplit)
    │
    ├─ Suivi MLflow ─────────► Expériences, métriques, artefacts, registre modèles
    │
    ├─ Évaluation ───────────► AUC-ROC, Precision@K, calibration
    │                          Explicabilité SHAP
    │                          Courbe de lift métier
    │
    └─ Déploiement ──────────► Endpoint REST FastAPI
                               Conteneur Docker
                               Script de scoring batch
```

## Résultats

| Modèle | AUC-ROC | Precision @ 10 % | Rappel |
|--------|---------|------------------|--------|
| Régression logistique (baseline) | 0,71 | 28 % | 0,41 |
| XGBoost | **0,84** | **42 %** | 0,58 |
| LightGBM | 0,83 | 40 % | 0,56 |

**Impact métier** : en ciblant les 15 % de clients les plus à risque, on capture 58 % des partants — les campagnes de rétention deviennent 4 fois plus rentables qu'un ciblage aléatoire.

## Variables clés (logique actuarielle)

- **Ratio prime / médiane de marché** — un contrat surfacturé déclenche plus de départs
- **Expérience sinistres** — une mauvaise gestion sinistre pousse au départ
- **Ancienneté + nombre de renouvellements** — indicateurs de fidélité
- **Évènements de vie (proxy)** — âge, canal d'acquisition, dernier contact
- **Concentration du portefeuille** — un client multi-équipement part moins (effet remise package)
- **Ratio S/P côté assureur** — les segments non rentables affichent des dynamiques de churn distinctes

## Stack technique

| Composant | Outil |
|-----------|-------|
| Framework ML | XGBoost, LightGBM, scikit-learn |
| Suivi des expériences | MLflow |
| Tuning des hyperparamètres | Optuna |
| Explicabilité | SHAP |
| API | FastAPI + Pydantic |
| Conteneurisation | Docker + Docker Compose |
| Tests | pytest + hypothesis |
| **CI/CD** | **GitHub Actions** (porte qualité AUC ≥ 0,70) |
| **Versionnage des données** | **DVC 3.x** |
| **Détection de dérive** | **Evidently AI** |
| **Qualité du code** | **ruff · black · isort · pre-commit** |
| **Gouvernance modèle** | **Model Card générée automatiquement** |

## Structure du projet

```
├── notebooks/
│   ├── 01_eda.ipynb                    # Analyse exploratoire des données
│   ├── 02_feature_engineering.ipynb    # Création et sélection des variables
│   ├── 03_model_training.ipynb         # Entraînement et évaluation
│   └── 04_shap_explainability.ipynb    # Analyse SHAP
├── src/
│   ├── data/
│   │   └── download_opendata.py        # Téléchargement + traduction au schéma assurance
│   ├── features/
│   │   ├── build_features.py           # Pipeline de features (entraînement / inférence)
│   │   └── actuarial_features.py       # Variables métier actuarielles
│   ├── models/
│   │   ├── train.py                    # Point d'entrée d'entraînement
│   │   ├── evaluate.py                 # Métriques et courbes de lift
│   │   └── predict.py                  # Utilitaires de scoring batch
│   ├── api/
│   │   ├── main.py                     # Application FastAPI
│   │   ├── schemas.py                  # Modèles Pydantic
│   │   └── utils.py                    # Helpers métier (CLV, risk tier, ...)
│   └── monitoring/
│       └── drift_detection.py          # Surveillance de dérive + Model Card
├── tests/
│   ├── test_features.py                # Tests du pipeline de features
│   ├── test_evaluate.py                # Tests des métriques d'évaluation
│   └── test_api.py                     # Tests de l'API
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── dvc.yaml                            # Pipeline reproductible DVC
├── params.yaml                         # Paramètres versionnés
└── Makefile                            # Raccourcis de commandes (`make data`, `make train`, ...)
```

## Démarrage rapide

```bash
git clone https://github.com/Abdousurf/insurance-churn-ml
cd insurance-churn-ml
pip install -r requirements.txt

# 1. Démarrer le serveur MLflow (utilisé par l'entraînement et l'API)
make mlflow                                    # http://127.0.0.1:5000

# 2. Télécharger le dataset public et le traduire au schéma du projet
make data                                      # écrit data/processed/*.parquet

# 3. Entraîner le modèle (logs MLflow, enregistré sous `insurance_churn_xgb`)
make train

# 4. Promouvoir la version entraînée au stage "Production"
python -c "import mlflow; \
  mlflow.set_tracking_uri('http://127.0.0.1:5000'); \
  mlflow.MlflowClient().transition_model_version_stage( \
    name='insurance_churn_xgb', version=1, stage='Production')"

# 5. Lancer l'API de prédiction
make api                                       # http://127.0.0.1:8000

# Ou avec Docker
docker-compose up
```

Tous les chemins MLflow / modèle sont configurables via variables
d'environnement (`MLFLOW_TRACKING_URI`, `MODEL_NAME`, `MODEL_STAGE`,
`FEATURE_BUILDER_PATH`) — des valeurs par défaut raisonnables sont
utilisées quand elles ne sont pas définies.

## Utilisation de l'API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "policy_id": "POL0012345",
    "tenure_months": 24,
    "annual_premium": 850,
    "claim_count_12m": 1,
    "lob": "auto",
    "renewal_count": 2,
    "channel": "Broker",
    "insured_age": 42
  }'

# Réponse :
{
  "policy_id": "POL0012345",
  "churn_probability": 0.34,
  "risk_tier": "medium",
  "recommended_action": "soft_retention",
  "estimated_clv": 4250.0,
  "model_version": "insurance_churn_xgb@Production"
}
```

La documentation interactive Swagger est disponible sur
`http://localhost:8000/docs` une fois l'API lancée.

---

*Construit en combinant connaissance actuarielle métier et bonnes pratiques d'ingénierie ML.*
*[LinkedIn](https://www.linkedin.com/in/abdou-john/)*
