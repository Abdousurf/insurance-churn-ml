"""Churn model training with hyperparameter optimization.

Trains XGBoost churn models using TimeSeriesSplit cross-validation,
Optuna hyperparameter tuning, MLflow experiment tracking, and SHAP
explainability. Includes a logistic regression baseline for comparison.
"""

import argparse
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import optuna
import shap
import warnings
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.features.actuarial_features import ActuarialFeatureBuilder
from src.models.evaluate import plot_lift_curve, plot_calibration, compute_business_metrics

warnings.filterwarnings("ignore")

MLFLOW_TRACKING_URI = "http://localhost:5000"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and split parquet data into features and target.

    Args:
        data_path: Path to the parquet file containing policy data.

    Returns:
        A tuple of (X, y) where X is the feature DataFrame (excluding
        churn_label and policy_id) and y is the churn_label Series.
    """
    df = pd.read_parquet(data_path)
    X = df.drop(columns=["churn_label", "policy_id"])
    y = df["churn_label"]
    return X, y


def objective_xgb(trial, X_train, y_train, cv):
    """Optuna objective function for XGBoost hyperparameter search.

    Args:
        trial: Optuna trial object for suggesting hyperparameters.
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        cv: Cross-validation splitter instance.

    Returns:
        Mean ROC-AUC score across cross-validation folds.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 5),
        "random_state": 42,
        "eval_metric": "auc",
        "verbosity": 0,
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def train_baseline(X_train, y_train):
    """Train a logistic regression baseline model.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.

    Returns:
        Fitted LogisticRegression model.
    """
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model


def train_with_optuna(X_train, y_train, n_trials: int = 50) -> XGBClassifier:
    """Train an XGBoost model with Optuna hyperparameter optimization.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        n_trials: Number of Optuna optimization trials to run.

    Returns:
        A tuple of (model, best_cv_auc) where model is the fitted
        XGBClassifier with best parameters and best_cv_auc is the
        best cross-validation AUC score achieved.
    """
    cv = TimeSeriesSplit(n_splits=5)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_xgb(trial, X_train, y_train, cv),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    best_params = study.best_params
    best_params.update({"random_state": 42, "eval_metric": "auc", "verbosity": 0})
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    return model, study.best_value


def main(args):
    """Run the full training pipeline with MLflow tracking.

    Trains a logistic baseline and an Optuna-tuned XGBoost model,
    logs metrics and artifacts to MLflow, and registers the best model.

    Args:
        args: Parsed command-line arguments with data_path,
            experiment_name, and n_trials attributes.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(args.experiment_name)

    print(f"Loading data from {args.data_path}...")
    X, y = load_data(Path(args.data_path))

    # Feature engineering
    builder = ActuarialFeatureBuilder()
    X_features = builder.fit_transform(X)

    # Train/test split (temporal — no shuffle!)
    split_idx = int(len(X_features) * 0.80)
    X_train, X_test = X_features.iloc[:split_idx], X_features.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Churn rate: {y.mean():.1%}")

    # ── Run 1: Baseline ─────────────────────────────────────────────────
    with mlflow.start_run(run_name="logistic_baseline"):
        baseline = train_baseline(X_train, y_train)
        y_prob_base = baseline.predict_proba(X_test)[:, 1]
        auc_base = roc_auc_score(y_test, y_prob_base)
        ap_base = average_precision_score(y_test, y_prob_base)
        brier_base = brier_score_loss(y_test, y_prob_base)

        mlflow.log_params({"model_type": "logistic_regression", "class_weight": "balanced"})
        mlflow.log_metrics({"auc_roc": auc_base, "avg_precision": ap_base, "brier_score": brier_base})
        mlflow.sklearn.log_model(baseline, "model")
        print(f"Baseline AUC: {auc_base:.4f}")

    # ── Run 2: XGBoost + Optuna ──────────────────────────────────────────
    with mlflow.start_run(run_name="xgboost_optuna"):
        print(f"Tuning XGBoost with {args.n_trials} Optuna trials...")
        xgb_model, best_cv_auc = train_with_optuna(X_train, y_train, args.n_trials)

        # Calibrate probabilities (Platt scaling)
        calibrated = CalibratedClassifierCV(xgb_model, method="sigmoid", cv="prefit")
        calibrated.fit(X_train, y_train)

        y_prob_xgb = calibrated.predict_proba(X_test)[:, 1]
        auc_xgb = roc_auc_score(y_test, y_prob_xgb)
        ap_xgb = average_precision_score(y_test, y_prob_xgb)
        brier_xgb = brier_score_loss(y_test, y_prob_xgb)

        business_metrics = compute_business_metrics(y_test, y_prob_xgb, top_k=0.15)

        mlflow.log_params({
            "model_type": "xgboost_calibrated",
            "n_trials": args.n_trials,
            **xgb_model.get_params()
        })
        mlflow.log_metrics({
            "auc_roc": auc_xgb,
            "cv_auc_best": best_cv_auc,
            "avg_precision": ap_xgb,
            "brier_score": brier_xgb,
            "lift_at_15pct": business_metrics["lift"],
            "recall_at_15pct": business_metrics["recall"],
            "precision_at_15pct": business_metrics["precision"],
        })

        # SHAP values
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        shap_fig = shap.summary_plot(shap_values, X_test, show=False)
        mlflow.log_figure(shap_fig, "shap_summary.png")

        # Lift curve
        lift_fig = plot_lift_curve(y_test, y_prob_xgb)
        mlflow.log_figure(lift_fig, "lift_curve.png")

        # Register best model
        mlflow.sklearn.log_model(
            calibrated, "model",
            registered_model_name="insurance_churn_xgb"
        )

        print(f"\n── Results ──────────────────────────────────────")
        print(f"  XGBoost AUC-ROC:      {auc_xgb:.4f}")
        print(f"  Lift at top 15%:      {business_metrics['lift']:.2f}x")
        print(f"  Recall at top 15%:    {business_metrics['recall']:.1%}")
        print(f"  Baseline AUC-ROC:     {auc_base:.4f}")
        print(f"  Improvement:          +{(auc_xgb - auc_base):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/churn_dataset.parquet")
    parser.add_argument("--experiment-name", default="churn_v1")
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()
    main(args)
