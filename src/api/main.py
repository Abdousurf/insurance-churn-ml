"""
Insurance Churn Prediction API
================================
FastAPI endpoint serving churn predictions from MLflow model registry.
Includes: prediction, batch scoring, health check, model info.
"""

from contextlib import asynccontextmanager
from typing import Optional
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import logging

from src.features.actuarial_features import ActuarialFeatureBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "insurance_churn_xgb"
MODEL_STAGE = "Production"

# ── Pydantic Schemas ────────────────────────────────────────────────────────

class PolicyFeatures(BaseModel):
    policy_id: str
    lob: str = Field(..., description="Line of Business: auto, home, liability, health")
    annual_premium: float = Field(..., gt=0, description="Annual premium in EUR")
    tenure_months: int = Field(..., ge=0, description="Policy tenure in months")
    renewal_count: int = Field(0, ge=0)
    claim_count_12m: int = Field(0, ge=0)
    claim_count_all: int = Field(0, ge=0)
    claim_settled_pct: float = Field(1.0, ge=0, le=1)
    days_to_settle_avg: float = Field(30.0, ge=0)
    insured_age: int = Field(..., ge=18, le=100)
    channel: str = Field(..., description="Acquisition channel: Direct, Broker, Online, Agent")
    policy_count_active: int = Field(1, ge=1)
    premium_change_pct: float = Field(0.0)
    last_contact_days: Optional[int] = None

    @field_validator("lob")
    @classmethod
    def validate_lob(cls, v):
        allowed = {"auto", "home", "liability", "health"}
        if v.lower() not in allowed:
            raise ValueError(f"lob must be one of {allowed}")
        return v.lower()

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v):
        allowed = {"Direct", "Broker", "Online", "Agent"}
        if v not in allowed:
            raise ValueError(f"channel must be one of {allowed}")
        return v


class ChurnPrediction(BaseModel):
    policy_id: str
    churn_probability: float
    risk_tier: str           # low / medium / high / critical
    recommended_action: str
    estimated_clv: float     # Customer Lifetime Value proxy (EUR)
    model_version: str


class BatchRequest(BaseModel):
    policies: list[PolicyFeatures]


class BatchResponse(BaseModel):
    predictions: list[ChurnPrediction]
    total_at_risk: int
    total_premium_at_risk: float


# ── Model loading ────────────────────────────────────────────────────────────

model = None
feature_builder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_builder
    logger.info(f"Loading model {MODEL_NAME}@{MODEL_STAGE} from MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        feature_builder = ActuarialFeatureBuilder()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Insurance Churn Prediction API",
    description="Predict which policyholders will churn — built by an actuarial data consultant.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────

RISK_TIERS = {
    "low": (0.0, 0.2),
    "medium": (0.2, 0.45),
    "high": (0.45, 0.70),
    "critical": (0.70, 1.0),
}

RETENTION_ACTIONS = {
    "low": "no_action",
    "medium": "soft_retention",   # loyalty email, mild offer
    "high": "proactive_outreach",  # call, 5-10% discount offer
    "critical": "urgent_retention", # personal advisor, max discount
}


def classify_risk(prob: float) -> str:
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= prob < hi:
            return tier
    return "critical"


def estimate_clv(features: PolicyFeatures) -> float:
    """Simple CLV proxy: premium × expected remaining tenure × discount factor."""
    expected_tenure_years = max(1, (60 - features.tenure_months) / 12)
    discount_rate = 0.08
    clv = features.annual_premium * (1 - (1 + discount_rate) ** -expected_tenure_years) / discount_rate
    return round(clv, 2)


def features_to_dataframe(policy: PolicyFeatures) -> pd.DataFrame:
    return pd.DataFrame([policy.model_dump()])


def predict_single(policy: PolicyFeatures) -> ChurnPrediction:
    df = features_to_dataframe(policy)
    df_features = feature_builder.transform(df)

    # Drop non-numeric / ID columns
    drop_cols = ["policy_id", "lob", "channel"]
    df_model = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns])

    prob = float(model.predict_proba(df_model)[:, 1][0])
    risk_tier = classify_risk(prob)

    return ChurnPrediction(
        policy_id=policy.policy_id,
        churn_probability=round(prob, 4),
        risk_tier=risk_tier,
        recommended_action=RETENTION_ACTIONS[risk_tier],
        estimated_clv=estimate_clv(policy),
        model_version=f"{MODEL_NAME}@{MODEL_STAGE}",
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "stage": MODEL_STAGE,
        "tracking_uri": MLFLOW_TRACKING_URI,
    }


@app.post("/predict", response_model=ChurnPrediction)
def predict(policy: PolicyFeatures):
    """Predict churn probability for a single policyholder."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predict_single(policy)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    """Batch churn predictions for a list of policies."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    predictions = [predict_single(p) for p in batch.policies]

    at_risk = [p for p in predictions if p.risk_tier in ("high", "critical")]
    premium_at_risk = sum(
        next(pol.annual_premium for pol in batch.policies if pol.policy_id == p.policy_id)
        for p in at_risk
    )

    return BatchResponse(
        predictions=predictions,
        total_at_risk=len(at_risk),
        total_premium_at_risk=round(premium_at_risk, 2),
    )
