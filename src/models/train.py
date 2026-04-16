"""Train a machine learning model to predict which insurance customers will leave.

This trains two models — a simple baseline and a more powerful one — then
picks the best settings automatically, tracks all results, and saves
the winning model for later use.
"""

# ───────────────────────────────────────────────────────
# WHAT THIS FILE DOES (in plain English):
# This file trains the churn prediction model. It:
# 1. Loads the insurance policy data
# 2. Creates useful features from that data
# 3. Trains a simple model first (as a comparison baseline)
# 4. Trains a smarter model and automatically finds the
#    best settings for it
# 5. Records all results so we can compare experiments
# 6. Saves the best model so it can be used for predictions
# ───────────────────────────────────────────────────────

import argparse
import mlflow
import mlflow.sklearn
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

from src.features.actuarial_features import ActuarialFeatureBuilder
from src.models.evaluate import plot_lift_curve, compute_business_metrics

# Suppress noisy warnings that clutter the output
warnings.filterwarnings("ignore")

# Where MLflow stores experiment results (must be running locally)
MLFLOW_TRACKING_URI = "http://localhost:5000"
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read the policy data file and separate features from the churn label.

    Args:
        data_path: Location of the data file on disk.

    Returns:
        A pair of (features, labels) — the input data and the "did they leave?"
        answer column, as separate items.
    """
    df = pd.read_parquet(data_path)
    X = df.drop(columns=["churn_label", "policy_id"])
    y = df["churn_label"]
    return X, y


def objective_xgb(trial, X_train, y_train, cv):
    """Try one combination of model settings and see how well it works.

    This is called many times by Optuna, each time with different settings,
    to find the best combination automatically.

    Args:
        trial: Optuna's way of suggesting different settings to try.
        X_train: The training data (features only).
        y_train: The training answers (did they churn?).
        cv: How to split the data for testing each combination.

    Returns:
        The average prediction quality score across all test splits.
    """
    # Let Optuna pick values for each setting within reasonable ranges
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
    # Train a model with these settings and measure how well it predicts
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def train_baseline(X_train, y_train):
    """Train a simple model as a baseline to compare the fancier model against.

    This uses logistic regression — a straightforward statistical model —
    so we know the minimum quality bar the advanced model should beat.

    Args:
        X_train: The training data (features only).
        y_train: The training answers (did they churn?).

    Returns:
        The trained simple model, ready to make predictions.
    """
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    return model


def train_with_optuna(X_train, y_train, n_trials: int = 50) -> XGBClassifier:
    """Train the main model while automatically finding the best settings.

    Optuna tries many different combinations of settings and keeps track
    of which ones work best. Then we train a final model using the
    winning combination.

    Args:
        X_train: The training data (features only).
        y_train: The training answers (did they churn?).
        n_trials: How many different setting combinations to try.

    Returns:
        A pair of (trained model, best score) — the final model using
        the best settings, and the best quality score achieved.
    """
    # Use time-based splits so we don't accidentally "peek into the future"
    cv = TimeSeriesSplit(n_splits=5)

    # Create a study that tries to find settings that maximize prediction quality
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective_xgb(trial, X_train, y_train, cv),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Take the best settings and train a final model with them
    best_params = study.best_params
    best_params.update({"random_state": 42, "eval_metric": "auc", "verbosity": 0})
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    return model, study.best_value


def main(args):
    """Run the complete training process from start to finish.

    This is the main workflow: load data, build features, train both models,
    measure their quality, record everything, and save the best model.

    Args:
        args: Command-line settings including the data file location,
            experiment name, and number of tuning trials to run.
    """
    # Connect to MLflow to track our experiment results
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(args.experiment_name)

    print(f"Loading data from {args.data_path}...")
    X, y = load_data(Path(args.data_path))

    # Turn raw data into useful features the model can learn from
    builder = ActuarialFeatureBuilder()
    X_features = builder.fit_transform(X)

    # Split into training and testing sets using time order (not random)
    # so we simulate real-world conditions where we predict the future
    split_idx = int(len(X_features) * 0.80)
    X_train, X_test = X_features.iloc[:split_idx], X_features.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Churn rate: {y.mean():.1%}")

    # ── Run 1: Baseline ─────────────────────────────────────────────────
    # Train the simple model first so we have something to compare against
    with mlflow.start_run(run_name="logistic_baseline"):
        baseline = train_baseline(X_train, y_train)
        y_prob_base = baseline.predict_proba(X_test)[:, 1]
        auc_base = roc_auc_score(y_test, y_prob_base)
        ap_base = average_precision_score(y_test, y_prob_base)
        brier_base = brier_score_loss(y_test, y_prob_base)

        # Record settings and results in MLflow
        mlflow.log_params({"model_type": "logistic_regression", "class_weight": "balanced"})
        mlflow.log_metrics({"auc_roc": auc_base, "avg_precision": ap_base, "brier_score": brier_base})
        mlflow.sklearn.log_model(baseline, "model")
        print(f"Baseline AUC: {auc_base:.4f}")

    # ── Run 2: XGBoost + Optuna ──────────────────────────────────────────
    # Train the advanced model with automatic setting optimization
    with mlflow.start_run(run_name="xgboost_optuna"):
        print(f"Tuning XGBoost with {args.n_trials} Optuna trials...")
        xgb_model, best_cv_auc = train_with_optuna(X_train, y_train, args.n_trials)

        # Adjust the model's probability outputs so they match real-world rates
        calibrated = CalibratedClassifierCV(xgb_model, method="sigmoid", cv="prefit")
        calibrated.fit(X_train, y_train)

        # Measure how well the model predicts on data it hasn't seen
        y_prob_xgb = calibrated.predict_proba(X_test)[:, 1]
        auc_xgb = roc_auc_score(y_test, y_prob_xgb)
        ap_xgb = average_precision_score(y_test, y_prob_xgb)
        brier_xgb = brier_score_loss(y_test, y_prob_xgb)

        # Calculate business-friendly metrics (e.g., "if we target the top 15%
        # of predicted churners, how many actual churners do we catch?")
        business_metrics = compute_business_metrics(y_test, y_prob_xgb, top_k=0.15)

        # Record all settings and results
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

        # Show which features matter most for predictions
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        shap_fig = shap.summary_plot(shap_values, X_test, show=False)
        mlflow.log_figure(shap_fig, "shap_summary.png")

        # Create a chart showing the model's advantage over random guessing
        lift_fig = plot_lift_curve(y_test, y_prob_xgb)
        mlflow.log_figure(lift_fig, "lift_curve.png")

        # Save the winning model so it can be loaded later for predictions
        mlflow.sklearn.log_model(
            calibrated, "model",
            registered_model_name="insurance_churn_xgb"
        )

        # Print a summary of how both models performed
        print("\n── Results ──────────────────────────────────────")
        print(f"  XGBoost AUC-ROC:      {auc_xgb:.4f}")
        print(f"  Lift at top 15%:      {business_metrics['lift']:.2f}x")
        print(f"  Recall at top 15%:    {business_metrics['recall']:.1%}")
        print(f"  Baseline AUC-ROC:     {auc_base:.4f}")
        print(f"  Improvement:          +{(auc_xgb - auc_base):.4f}")


# This section runs only when you execute this file directly from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/churn_dataset.parquet")
    parser.add_argument("--experiment-name", default="churn_v1")
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()
    main(args)
