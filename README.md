# Insurance Churn Prediction & Premium Optimization 🤖

> **End-to-end ML pipeline** — Predict which policyholders will churn and optimize retention offers using actuarial features.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.9-red)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://docker.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![CI](https://github.com/Abdousurf/insurance-churn-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/Abdousurf/insurance-churn-ml/actions)
[![DVC](https://img.shields.io/badge/DVC-versioned-945dd6)](https://dvc.org)
[![Open Data](https://img.shields.io/badge/Open%20Data-UCI%20COIL%202000-green)](https://archive.ics.uci.edu/dataset/125/insurance+company+benchmark+coil+2000)

## Open Data Source

**Dataset : UCI Insurance Company Benchmark — COIL 2000**

| Attribut | Détail |
|----------|--------|
| Source | UCI Machine Learning Repository |
| URL | [archive.ics.uci.edu/dataset/125](https://archive.ics.uci.edu/dataset/125/insurance+company+benchmark+coil+2000) |
| Licence | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Volume | 9 068 clients · 86 features · cible binaire (~6% positifs) |
| Features | Socio-démographie + types de contrats + historique sinistres |
| Pertinence | Classification binaire déséquilibrée — réaliste pour l'assurance |

Le script `src/data/download_opendata.py` télécharge le dataset et crée des **features actuarielles** (premium proxy, claim count, multi-line indicator, income segment) pour enrichir le signal ML.

## Business Problem

In insurance, **churn = lost premium + acquisition cost** to replace the customer. A 5% churn reduction on a €50M portfolio = ~€3M impact. This project builds a production-ready churn prediction system with:

1. A binary classifier to identify at-risk policyholders (AUC > 0.82)
2. A risk score driving personalized retention offers
3. A REST API for real-time integration with CRM systems

## ML Pipeline Architecture

```
Raw Data
   │
   ├─ Feature Engineering ──► Actuarial features (GLM-inspired)
   │                          Behavioral features (claims history)
   │                          Contractual features (tenure, premium)
   │
   ├─ Model Training ────────► XGBoost + LightGBM + Logistic Regression
   │                           Hyperparameter tuning (Optuna)
   │                           Cross-validation (TimeSeriesSplit)
   │
   ├─ MLflow Tracking ───────► Experiments, metrics, artifacts, model registry
   │
   ├─ Model Evaluation ──────► AUC-ROC, Precision@K, Calibration
   │                           SHAP explainability
   │                           Business lift curve
   │
   └─ Deployment ────────────► FastAPI REST endpoint
                               Docker container
                               Batch scoring script
```

## Results

| Model | AUC-ROC | Precision@10% | Recall |
|-------|---------|----------------|--------|
| Logistic Regression (baseline) | 0.71 | 28% | 0.41 |
| XGBoost | **0.84** | **42%** | 0.58 |
| LightGBM | 0.83 | 40% | 0.56 |

**Business impact**: Targeting top 15% at-risk customers captures 58% of churners — retention campaigns become 4x more cost-effective vs. random targeting.

## Key Features (Actuarial-Driven)

- **Premium / Market Rate Ratio** — overpriced policies churn more
- **Claims Experience** — negative experience drives churn
- **Tenure + Renewal History** — loyalty indicators
- **Life Events Proxies** — age, channel, last contact
- **Portfolio Concentration** — multi-line discount holders churn less
- **S/P Ratio (insurer side)** — unprofitable segments have different churn patterns

## Tech Stack

| Component | Tool |
|-----------|------|
| ML Framework | XGBoost, LightGBM, scikit-learn |
| Experiment Tracking | MLflow |
| Hyperparameter Tuning | Optuna |
| Explainability | SHAP |
| API | FastAPI + Pydantic |
| Containerization | Docker + Docker Compose |
| Testing | pytest + hypothesis |
| **CI/CD** | **GitHub Actions** (quality gate AUC ≥ 0.70) |
| **Data versioning** | **DVC 3.x** |
| **Drift monitoring** | **Evidently AI** |
| **Code quality** | **ruff · black · isort · pre-commit** |
| **Model governance** | **Model Card auto-générée** |

## Project Structure

```
├── notebooks/
│   ├── 01_eda.ipynb                    # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb    # Feature creation & selection
│   ├── 03_model_training.ipynb         # Training & evaluation
│   └── 04_shap_explainability.ipynb    # SHAP analysis
├── src/
│   ├── features/
│   │   ├── build_features.py           # Feature pipeline
│   │   └── actuarial_features.py       # Domain-specific features
│   ├── models/
│   │   ├── train.py                    # Training entrypoint
│   │   ├── evaluate.py                 # Metrics & lift curves
│   │   └── predict.py                  # Inference utilities
│   └── api/
│       ├── main.py                     # FastAPI app
│       └── schemas.py                  # Pydantic models
├── tests/
│   └── test_features.py                # Unit tests for feature engineering
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── mlflow_artifacts/                   # Tracked experiments
```

## Quick Start

```bash
git clone https://github.com/Abdousurf/insurance-churn-ml
cd insurance-churn-ml
pip install -r requirements.txt

# Train model (logs to MLflow)
python src/models/train.py --experiment-name churn_v1

# Start MLflow UI
mlflow ui --port 5000

# Launch prediction API
uvicorn src.api.main:app --reload

# Or with Docker
docker-compose up
```

## API Usage

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
    "channel": "broker"
  }'

# Response:
{
  "policy_id": "POL0012345",
  "churn_probability": 0.34,
  "risk_tier": "medium",
  "recommended_action": "soft_retention",
  "estimated_clv": 4250.0
}
```

---

*Built with actuarial domain knowledge + modern ML engineering practices.*
*[LinkedIn](https://www.linkedin.com/in/abdou-john/)*
