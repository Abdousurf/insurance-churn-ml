import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
    calibration_curve,
)


def compute_business_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    top_k: float = 0.15,
) -> dict:
    n = len(y_true)
    k = int(n * top_k)

    sorted_idx = np.argsort(y_prob)[::-1]
    y_sorted = np.asarray(y_true)[sorted_idx]

    churners_in_top_k = y_sorted[:k].sum()
    total_churners = np.asarray(y_true).sum()

    precision_at_k = churners_in_top_k / k if k > 0 else 0.0
    recall_at_k = churners_in_top_k / total_churners if total_churners > 0 else 0.0
    base_rate = total_churners / n if n > 0 else 0.0
    lift = precision_at_k / base_rate if base_rate > 0 else 0.0

    return {
        "precision": round(precision_at_k, 4),
        "recall": round(recall_at_k, 4),
        "lift": round(lift, 2),
        "top_k": top_k,
        "churners_captured": int(churners_in_top_k),
        "total_churners": int(total_churners),
    }


def plot_lift_curve(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
    n = len(y_true)
    sorted_idx = np.argsort(y_prob)[::-1]
    y_sorted = np.asarray(y_true)[sorted_idx]

    percentiles = np.arange(1, n + 1) / n
    cumulative_churners = np.cumsum(y_sorted)
    total_churners = y_sorted.sum()
    cumulative_recall = cumulative_churners / total_churners

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(percentiles * 100, cumulative_recall * 100, label="Model", linewidth=2)
    ax.plot([0, 100], [0, 100], "--", color="gray", label="Random")
    ax.set_xlabel("% Population Targeted")
    ax.set_ylabel("% Churners Captured")
    ax.set_title("Lift Curve — Cumulative Gains")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> plt.Figure:
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def full_evaluation_report(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "auc_roc": round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision": round(average_precision_score(y_true, y_prob), 4),
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
        "business_metrics_10pct": compute_business_metrics(y_true, y_prob, top_k=0.10),
        "business_metrics_15pct": compute_business_metrics(y_true, y_prob, top_k=0.15),
    }
