"""Measure how well the churn model performs, using both technical and business-friendly metrics.

Provides ways to check model quality from a business perspective (e.g., "if we
contact the top 15% riskiest customers, how many actual churners do we catch?")
and creates charts to visualize the results.
"""

# ───────────────────────────────────────────────────────
# WHAT THIS FILE DOES (in plain English):
# This file measures how good the churn prediction model
# is. It calculates scores that tell us things like:
# - How well the model ranks customers by churn risk
# - If we focus on the riskiest customers, how many real
#   churners do we actually catch?
# - Are the model's probability estimates trustworthy?
# It also creates charts to visualize these results.
# ───────────────────────────────────────────────────────

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    roc_curve,
    calibration_curve,
)


def compute_business_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    top_k: float = 0.15,
) -> dict:
    """Calculate business-friendly performance metrics for a given target group.

    Answers the question: "If we focus our retention efforts on the top X%
    riskiest customers, how effective would that be?"

    Args:
        y_true: The actual outcomes — did each customer really leave? (0 or 1)
        y_prob: The model's predicted chance of each customer leaving (0 to 1).
        top_k: What fraction of customers to target (e.g., 0.15 means top 15%).

    Returns:
        A dictionary with: how precise our targeting is, what fraction of
        churners we'd catch, how much better than random this is (lift),
        and raw counts.
    """
    n = len(y_true)
    # Figure out how many customers fall in the "top k%" group
    k = int(n * top_k)

    # Sort customers from highest predicted risk to lowest
    sorted_idx = np.argsort(y_prob)[::-1]
    y_sorted = np.asarray(y_true)[sorted_idx]

    # Count how many actual churners are in our target group
    churners_in_top_k = y_sorted[:k].sum()
    total_churners = np.asarray(y_true).sum()

    # What percentage of our target group are actual churners?
    precision_at_k = churners_in_top_k / k if k > 0 else 0.0
    # What percentage of all churners did we catch?
    recall_at_k = churners_in_top_k / total_churners if total_churners > 0 else 0.0
    # How much better is this than picking customers at random?
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
    """Create a chart showing how much better the model is than random guessing.

    The lift curve shows: as we contact more and more customers (starting
    from the riskiest), what percentage of all churners have we found?
    A good model finds most churners quickly; random guessing follows a
    straight diagonal line.

    Args:
        y_true: The actual outcomes — did each customer really leave? (0 or 1)
        y_prob: The model's predicted chance of each customer leaving (0 to 1).

    Returns:
        A chart (matplotlib Figure) showing the lift curve.
    """
    n = len(y_true)
    # Sort customers from highest predicted risk to lowest
    sorted_idx = np.argsort(y_prob)[::-1]
    y_sorted = np.asarray(y_true)[sorted_idx]

    # Calculate running totals as we go down the list
    percentiles = np.arange(1, n + 1) / n
    cumulative_churners = np.cumsum(y_sorted)
    total_churners = y_sorted.sum()
    cumulative_recall = cumulative_churners / total_churners

    # Draw the chart
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
    """Create a chart showing whether the model's probability estimates are accurate.

    If the model says "30% chance of churning" for a group of customers,
    do roughly 30% of them actually churn? This chart checks that.

    Args:
        y_true: The actual outcomes — did each customer really leave? (0 or 1)
        y_prob: The model's predicted chance of each customer leaving (0 to 1).
        n_bins: How many groups to split the predictions into for comparison.

    Returns:
        A chart (matplotlib Figure) showing predicted vs. actual probabilities.
    """
    # Split predictions into groups and compare predicted vs. actual rates
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    # Draw the chart — a perfect model follows the diagonal line
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
    """Create a chart showing the model's ability to tell churners from non-churners.

    The ROC curve shows the trade-off between catching more churners (good)
    and falsely flagging non-churners (bad). The area under the curve (AUC)
    gives a single score — closer to 1.0 is better, 0.5 is random guessing.

    Args:
        y_true: The actual outcomes — did each customer really leave? (0 or 1)
        y_prob: The model's predicted chance of each customer leaving (0 to 1).

    Returns:
        A chart (matplotlib Figure) with the ROC curve and its AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    # Draw the chart
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
    """Create a complete quality report with all key metrics in one place.

    Args:
        y_true: The actual outcomes — did each customer really leave? (0 or 1)
        y_prob: The model's predicted chance of each customer leaving (0 to 1).

    Returns:
        A dictionary with the overall quality score (AUC), precision score,
        calibration score, and business metrics at the 10% and 15% targeting levels.
    """
    return {
        "auc_roc": round(roc_auc_score(y_true, y_prob), 4),
        "avg_precision": round(average_precision_score(y_true, y_prob), 4),
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
        "business_metrics_10pct": compute_business_metrics(y_true, y_prob, top_k=0.10),
        "business_metrics_15pct": compute_business_metrics(y_true, y_prob, top_k=0.15),
    }
