"""Tests for the model evaluation metrics and visualization functions."""

import numpy as np
import pytest

from src.models.evaluate import (
    compute_business_metrics,
    full_evaluation_report,
    plot_lift_curve,
    plot_roc_curve,
    plot_calibration,
)


class TestComputeBusinessMetrics:
    """Tests for business metrics calculation."""

    @pytest.fixture
    def perfect_predictions(self):
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        return y_true, y_prob

    def test_perfect_model_high_lift(self, perfect_predictions):
        y_true, y_prob = perfect_predictions
        metrics = compute_business_metrics(y_true, y_prob, top_k=0.30)
        assert metrics["lift"] > 1.0
        assert metrics["recall"] == 1.0

    def test_keys_present(self, perfect_predictions):
        y_true, y_prob = perfect_predictions
        metrics = compute_business_metrics(y_true, y_prob)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "lift" in metrics
        assert "churners_captured" in metrics
        assert "total_churners" in metrics

    def test_all_zeros(self):
        y_true = np.array([0, 0, 0, 0, 0])
        y_prob = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        metrics = compute_business_metrics(y_true, y_prob)
        assert metrics["total_churners"] == 0

    def test_recall_between_0_and_1(self, perfect_predictions):
        y_true, y_prob = perfect_predictions
        metrics = compute_business_metrics(y_true, y_prob, top_k=0.10)
        assert 0.0 <= metrics["recall"] <= 1.0


class TestPlotFunctions:
    """Tests for visualization functions."""

    @pytest.fixture
    def sample_data(self):
        y_true = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.7, 0.3, 0.2, 0.1, 0.8, 0.4, 0.15, 0.05, 0.25])
        return y_true, y_prob

    def test_lift_curve_returns_figure(self, sample_data):
        import matplotlib.pyplot as plt

        fig = plot_lift_curve(*sample_data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_roc_curve_returns_figure(self, sample_data):
        import matplotlib.pyplot as plt

        fig = plot_roc_curve(*sample_data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_calibration_returns_figure(self, sample_data):
        import matplotlib.pyplot as plt

        fig = plot_calibration(*sample_data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestFullEvaluationReport:
    """Tests for the full evaluation report."""

    def test_report_keys(self):
        y_true = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.7, 0.3, 0.2, 0.1, 0.8, 0.4, 0.15, 0.05, 0.25])
        report = full_evaluation_report(y_true, y_prob)
        assert "auc_roc" in report
        assert "avg_precision" in report
        assert "brier_score" in report
        assert "business_metrics_10pct" in report
        assert "business_metrics_15pct" in report

    def test_auc_between_0_and_1(self):
        y_true = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.7, 0.3, 0.2, 0.1, 0.8, 0.4, 0.15, 0.05, 0.25])
        report = full_evaluation_report(y_true, y_prob)
        assert 0.0 <= report["auc_roc"] <= 1.0
