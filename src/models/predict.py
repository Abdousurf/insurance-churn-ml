"""Prediction and batch scoring for insurance churn models.

Loads production models from MLflow and provides single-DataFrame and
batch file scoring with risk tier classification.
"""

import mlflow
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.actuarial_features import ActuarialFeatureBuilder
from src.features.build_features import NON_FEATURE_COLS


MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "insurance_churn_xgb"
MODEL_STAGE = "Production"


def load_production_model():
    """Load the production churn model from MLflow model registry.

    Returns:
        The registered production sklearn model.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    return model


def predict_dataframe(
    df: pd.DataFrame,
    model,
    builder: ActuarialFeatureBuilder,
) -> pd.DataFrame:
    """Score a DataFrame and return churn probabilities with risk tiers.

    Args:
        df: Raw policy DataFrame to score.
        model: Fitted sklearn-compatible model with predict_proba.
        builder: Pre-fitted ActuarialFeatureBuilder instance.

    Returns:
        DataFrame with policy_id (if present), churn_probability,
        and risk_tier columns.
    """
    df_features = builder.transform(df)
    feature_cols = [c for c in df_features.columns if c not in NON_FEATURE_COLS]
    X = df_features[feature_cols]

    probas = model.predict_proba(X)[:, 1]

    result = df[["policy_id"]].copy() if "policy_id" in df.columns else pd.DataFrame()
    result["churn_probability"] = probas
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
    """Score a parquet file of policies and write results to disk.

    Args:
        input_path: Path to the input parquet file.
        output_path: Path to write the scored output parquet file.
        model: Optional pre-loaded model. If None, loads from MLflow.
        builder: Optional pre-fitted feature builder. If None, creates
            a new one and fits on the input data.

    Returns:
        DataFrame with scoring results.
    """
    if model is None:
        model = load_production_model()
    if builder is None:
        builder = ActuarialFeatureBuilder()

    df = pd.read_parquet(input_path)
    builder.fit(df)
    results = predict_dataframe(df, model, builder)
    results.to_parquet(output_path, index=False)
    print(f"Scored {len(results)} policies → {output_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch scoring")
    parser.add_argument("--input", required=True, help="Path to input parquet")
    parser.add_argument("--output", default="data/scores.parquet")
    args = parser.parse_args()

    batch_score(Path(args.input), Path(args.output))
