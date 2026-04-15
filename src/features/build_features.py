"""Turn raw insurance policy data into a set of useful numbers the model can learn from.

Provides functions to build training and prediction-ready data tables
from raw policy records using insurance-specific calculations.
"""

# ───────────────────────────────────────────────────────
# WHAT THIS FILE DOES (in plain English):
# This file reads raw insurance policy data and turns it
# into a table of meaningful numbers (called "features")
# that the churn prediction model can use to learn patterns.
# It handles both training (learning) and live prediction
# scenarios.
# ───────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from pathlib import Path

from src.features.actuarial_features import ActuarialFeatureBuilder


# These columns are identifiers or raw text — not useful as model inputs,
# so we exclude them from the final feature set.
NON_FEATURE_COLS = [
    "policy_id", "customer_id", "inception_date", "expiry_date",
    "lob", "region", "channel",
    "churn_label",
]


def build_training_features(
    data_path: Path,
    output_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Read raw policy data from a file and create the training-ready feature table.

    This loads a data file, separates the "did this customer leave?" label
    from the rest of the data, then creates all the insurance-specific
    calculated columns the model needs to learn from.

    Args:
        data_path: Where the raw policy data file lives on disk.
        output_path: If provided, saves the finished feature table to this
            location. If not provided, nothing is saved to disk.

    Returns:
        A pair of (features, labels) where features is the table of numbers
        the model learns from, and labels tells us which customers actually left.
    """
    # Load the raw data from a file
    df = pd.read_parquet(data_path)

    # Separate the answer column ("did they churn?") from the rest
    y = df["churn_label"]
    X = df.drop(columns=["churn_label"])

    # Create the feature builder and generate all insurance-specific columns
    builder = ActuarialFeatureBuilder()
    X_features = builder.fit_transform(X)

    # Keep only the columns that are actual features (drop IDs, text fields, etc.)
    feature_cols = [c for c in X_features.columns if c not in NON_FEATURE_COLS]
    X_final = X_features[feature_cols]

    # Optionally save the results to disk for later use
    if output_path is not None:
        out = X_final.copy()
        out["churn_label"] = y.values
        out.to_parquet(output_path, index=False)

    return X_final, y


def build_inference_features(df: pd.DataFrame, builder: ActuarialFeatureBuilder) -> pd.DataFrame:
    """Create the feature table for making predictions on new data.

    Unlike training, this uses a builder that has already learned from
    past data so the new data gets processed the same way.

    Args:
        df: Raw policy data to prepare for prediction.
        builder: A feature builder that has already been set up on training data.

    Returns:
        A table containing only the model-ready feature columns.
    """
    # Apply the same transformations used during training
    df_features = builder.transform(df)

    # Keep only the feature columns, drop IDs and text fields
    feature_cols = [c for c in df_features.columns if c not in NON_FEATURE_COLS]
    return df_features[feature_cols]


# This section runs only when you execute this file directly from the command line
if __name__ == "__main__":
    import argparse

    # Set up command-line options so the user can specify input/output files
    parser = argparse.ArgumentParser(description="Build feature matrix from raw data")
    parser.add_argument("--input", default="data/churn_dataset.parquet")
    parser.add_argument("--output", default="data/features.parquet")
    args = parser.parse_args()

    # Run the feature building and print a quick summary
    X, y = build_training_features(Path(args.input), Path(args.output))
    print(f"Feature matrix: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Churn rate: {y.mean():.1%}")
