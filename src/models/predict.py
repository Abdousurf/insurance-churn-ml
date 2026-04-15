"""Make churn predictions on new insurance policies using the trained model.

Loads the saved model and scores individual policies or entire files,
assigning each customer a risk level (low, medium, high, critical).
"""

# ───────────────────────────────────────────────────────
# WHAT THIS FILE DOES (in plain English):
# This file uses the trained model to predict which
# customers are likely to leave. It can:
# - Load the saved production model
# - Score one batch of customer data at a time
# - Assign each customer a risk level (low/medium/high/critical)
# - Save the results to a file
# ───────────────────────────────────────────────────────

import mlflow
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.actuarial_features import ActuarialFeatureBuilder
from src.features.build_features import NON_FEATURE_COLS


# Where to find MLflow and which model to load
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "insurance_churn_xgb"
MODEL_STAGE = "Production"


def load_production_model():
    """Load the live production model that's been saved in MLflow.

    Returns:
        The trained model, ready to make predictions.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    return model


def predict_dataframe(
    df: pd.DataFrame,
    model,
    builder: ActuarialFeatureBuilder,
) -> pd.DataFrame:
    """Predict churn risk for a table of customers and assign risk levels.

    Each customer gets a probability (0-100% chance of leaving) and a
    risk category: low, medium, high, or critical.

    Args:
        df: Raw policy data for the customers to score.
        model: The trained prediction model.
        builder: A feature builder that's already been set up on training data.

    Returns:
        A table with each customer's policy ID (if available), their predicted
        churn probability, and their risk level.
    """
    # Turn raw data into the features the model expects
    df_features = builder.transform(df)
    feature_cols = [c for c in df_features.columns if c not in NON_FEATURE_COLS]
    X = df_features[feature_cols]

    # Get the model's predicted probability of churn for each customer
    probas = model.predict_proba(X)[:, 1]

    # Build the results table with risk tiers
    result = df[["policy_id"]].copy() if "policy_id" in df.columns else pd.DataFrame()
    result["churn_probability"] = probas
    # Assign a risk level based on the probability
    result["risk_tier"] = pd.cut(
        probas,
        bins=[0, 0.2, 0.45, 0.70, 1.0],
        labels=["low", "medium", "high", "critical"],
        include_lowest=True,
    )
    return result


def batch_score(
    input_path: Path,
    output_path: Path,
    model=None,
    builder: ActuarialFeatureBuilder | None = None,
):
    """Score an entire file of policies and save the results.

    Reads a file of customer data, predicts churn risk for each one,
    and writes the results to a new file.

    Args:
        input_path: Where the customer data file is located.
        output_path: Where to save the scored results.
        model: The trained model. If not provided, loads the production model automatically.
        builder: The feature builder. If not provided, creates a new one from the data.

    Returns:
        A table with the scoring results for all customers.
    """
    # Load the model if one wasn't provided
    if model is None:
        model = load_production_model()
    # Create a feature builder if one wasn't provided
    if builder is None:
        builder = ActuarialFeatureBuilder()

    # Read the data, build features, make predictions, and save
    df = pd.read_parquet(input_path)
    builder.fit(df)
    results = predict_dataframe(df, model, builder)
    results.to_parquet(output_path, index=False)
    print(f"Scored {len(results)} policies → {output_path}")
    return results


# This section runs only when you execute this file directly from the command line
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch scoring")
    parser.add_argument("--input", required=True, help="Path to input parquet")
    parser.add_argument("--output", default="data/scores.parquet")
    args = parser.parse_args()

    batch_score(Path(args.input), Path(args.output))
