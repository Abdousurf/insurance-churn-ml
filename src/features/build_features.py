import pandas as pd
import numpy as np
from pathlib import Path

from src.features.actuarial_features import ActuarialFeatureBuilder


NON_FEATURE_COLS = [
    "policy_id", "customer_id", "inception_date", "expiry_date",
    "lob", "region", "channel",
    "churn_label",
]


def build_training_features(
    data_path: Path,
    output_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(data_path)

    y = df["churn_label"]
    X = df.drop(columns=["churn_label"])

    builder = ActuarialFeatureBuilder()
    X_features = builder.fit_transform(X)

    feature_cols = [c for c in X_features.columns if c not in NON_FEATURE_COLS]
    X_final = X_features[feature_cols]

    if output_path is not None:
        out = X_final.copy()
        out["churn_label"] = y.values
        out.to_parquet(output_path, index=False)

    return X_final, y


def build_inference_features(df: pd.DataFrame, builder: ActuarialFeatureBuilder) -> pd.DataFrame:
    df_features = builder.transform(df)
    feature_cols = [c for c in df_features.columns if c not in NON_FEATURE_COLS]
    return df_features[feature_cols]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build feature matrix from raw data")
    parser.add_argument("--input", default="data/churn_dataset.parquet")
    parser.add_argument("--output", default="data/features.parquet")
    args = parser.parse_args()

    X, y = build_training_features(Path(args.input), Path(args.output))
    print(f"Feature matrix: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Churn rate: {y.mean():.1%}")
