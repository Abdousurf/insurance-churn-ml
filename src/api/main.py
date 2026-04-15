"""Insurance churn prediction API.

FastAPI application serving churn predictions from the MLflow model registry.
Provides single and batch prediction endpoints, health checks, and model info.
"""

from contextlib import asynccontextmanager
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from src.features.actuarial_features import ActuarialFeatureBuilder
from src.api.schemas import PolicyFeatures, ChurnPrediction, BatchRequest, BatchResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "insurance_churn_xgb"
MODEL_STAGE = "Production"


# ── Model loading ────────────────────────────────────────────────────────────

model = None
feature_builder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan: load model on startup, cleanup on shutdown.

    Args:
        app: The FastAPI application instance.

    Yields:
        None after model is loaded and ready to serve.

    Raises:
        Exception: If the model fails to load from MLflow.
    """
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
    """Classify a churn probability into a risk tier.

    Args:
        prob: Churn probability between 0 and 1.

    Returns:
        Risk tier string: "low", "medium", "high", or "critical".
    """
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= prob < hi:
            return tier
    return "critical"


def estimate_clv(features: PolicyFeatures) -> float:
    """Estimate customer lifetime value using a discounted cash flow proxy.

    Args:
        features: PolicyFeatures instance with annual_premium and tenure_months.

    Returns:
        Estimated CLV in EUR, rounded to 2 decimal places.
    """
    expected_tenure_years = max(1, (60 - features.tenure_months) / 12)
    discount_rate = 0.08
    clv = features.annual_premium * (1 - (1 + discount_rate) ** -expected_tenure_years) / discount_rate
    return round(clv, 2)


def features_to_dataframe(policy: PolicyFeatures) -> pd.DataFrame:
    """Convert a PolicyFeatures instance to a single-row DataFrame.

    Args:
        policy: PolicyFeatures Pydantic model instance.

    Returns:
        Single-row DataFrame suitable for model inference.
    """
    return pd.DataFrame([policy.model_dump()])


def predict_single(policy: PolicyFeatures) -> ChurnPrediction:
    """Generate a churn prediction for a single policyholder.

    Args:
        policy: PolicyFeatures with the policyholder's attributes.

    Returns:
        ChurnPrediction with probability, risk tier, recommended action,
        estimated CLV, and model version.
    """
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
    """Check API health and model availability.

    Returns:
        Dict with status and model_loaded flag.
    """
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    """Return metadata about the currently loaded model.

    Returns:
        Dict with model_name, stage, and tracking_uri.
    """
    return {
        "model_name": MODEL_NAME,
        "stage": MODEL_STAGE,
        "tracking_uri": MLFLOW_TRACKING_URI,
    }


@app.post("/predict", response_model=ChurnPrediction)
def predict(policy: PolicyFeatures):
    """Predict churn probability for a single policyholder.

    Args:
        policy: PolicyFeatures request body with policyholder attributes.

    Returns:
        ChurnPrediction with probability, risk tier, and recommended action.

    Raises:
        HTTPException: 503 if the model is not loaded.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predict_single(policy)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    """Generate batch churn predictions for multiple policies.

    Args:
        batch: BatchRequest containing a list of PolicyFeatures.

    Returns:
        BatchResponse with individual predictions and aggregate risk summary.

    Raises:
        HTTPException: 503 if the model is not loaded.
    """
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
