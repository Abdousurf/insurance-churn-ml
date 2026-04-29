"""Web API for insurance churn predictions.

A web service that accepts customer data and returns churn risk predictions.
Supports single customer lookups and batch scoring of many customers at once.
"""

# ───────────────────────────────────────────────────────
# WHAT THIS FILE DOES (in plain English):
# This file runs a web server (API) that lets other
# applications send customer data and get back churn
# predictions. It supports:
# - Checking if the service is running (/health)
# - Getting model details (/model-info)
# - Predicting churn for one customer (/predict)
# - Predicting churn for many customers at once (/predict/batch)
# ───────────────────────────────────────────────────────

from contextlib import asynccontextmanager
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

from src.features.actuarial_features import ActuarialFeatureBuilder
from src.api.schemas import PolicyFeatures, ChurnPrediction, BatchRequest, BatchResponse
from src.api.utils import (
    classify_risk,
    estimate_clv,
    features_to_dataframe,
    RETENTION_ACTIONS,
)

# Set up logging so we can see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Where to find MLflow and which model to load
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "insurance_churn_xgb"
MODEL_STAGE = "Production"


# ── Model loading ────────────────────────────────────────────────────────────

# These hold the model and feature builder once the server starts up
model = None
feature_builder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the prediction model when the server starts, clean up when it stops.

    Args:
        app: The web application.

    Yields:
        Nothing — just pauses between startup and shutdown.

    Raises:
        Exception: If the model can't be loaded from MLflow.
    """
    global model, feature_builder
    logger.info(f"Loading model {MODEL_NAME}@{MODEL_STAGE} from MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        # Load the trained model so it's ready to make predictions
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
        feature_builder = ActuarialFeatureBuilder()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    logger.info("Shutting down")


# Create the web application
app = FastAPI(
    title="Insurance Churn Prediction API",
    description="Predict which policyholders will churn — built by an actuarial data consultant.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from any website (needed for web frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def predict_single(policy: PolicyFeatures) -> ChurnPrediction:
    """Get a full churn prediction for one customer.

    Takes the customer's details, runs them through the model, and returns
    the churn probability, risk level, what action to take, and how much
    the customer is worth.

    Args:
        policy: The customer's policy details.

    Returns:
        A complete prediction with probability, risk tier, recommended action,
        estimated customer value, and which model version was used.
    """
    # Convert the customer data to a table and add insurance-specific features
    df = features_to_dataframe(policy)
    df_features = feature_builder.transform(df)

    # Remove columns the model doesn't use (IDs and text fields)
    drop_cols = ["policy_id", "lob", "channel"]
    df_model = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns])

    # Get the model's prediction
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
    """Check if the API is running and the model is loaded.

    Returns:
        Status information showing whether the service is healthy.
    """
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info")
def model_info():
    """Get details about which model is currently loaded.

    Returns:
        The model's name, stage, and where it's stored.
    """
    return {
        "model_name": MODEL_NAME,
        "stage": MODEL_STAGE,
        "tracking_uri": MLFLOW_TRACKING_URI,
    }


@app.post("/predict", response_model=ChurnPrediction)
def predict(policy: PolicyFeatures):
    """Predict churn risk for a single customer.

    Send in a customer's details and get back their churn probability,
    risk level, and what retention action is recommended.

    Args:
        policy: The customer's policy details sent in the request.

    Returns:
        The churn prediction with probability, risk tier, and recommended action.

    Raises:
        HTTPException: 503 error if the model hasn't been loaded yet.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predict_single(policy)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    """Predict churn risk for multiple customers at once.

    Send in a list of customers and get back individual predictions for
    each one, plus a summary of how many are at risk and how much
    premium revenue could be lost.

    Args:
        batch: A list of customer policy details.

    Returns:
        Individual predictions for each customer, plus a risk summary.

    Raises:
        HTTPException: 503 error if the model hasn't been loaded yet.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Score each customer individually
    predictions = [predict_single(p) for p in batch.policies]

    # Summarize: how many customers are high risk, and how much revenue is at stake?
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
