"""Tests des métriques d'évaluation et des fonctions de visualisation."""

# ───────────────────────────────────────────────────────
# RÔLE DE CE FICHIER (en clair) :
# Il vérifie que les calculs de "qualité du modèle"
# (AUC, lift, courbes…) renvoient des valeurs cohérentes :
# entre 0 et 1, supérieures au tirage aléatoire pour un
# modèle parfait, et que les graphiques sont bien produits.
# ───────────────────────────────────────────────────────

import numpy as np
import pytest

from src.models import (
    compute_business_metrics,
    full_evaluation_report,
    plot_calibration,
    plot_lift_curve,
    plot_roc_curve,
)


class TestComputeBusinessMetrics:
    """Tests du calcul des métriques métier (lift, recall, precision)."""

    @pytest.fixture
    def perfect_predictions(self):
        """Renvoie un cas où le modèle classe parfaitement les 3 partants en tête."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        return y_true, y_prob

    def test_perfect_model_high_lift(self, perfect_predictions):
        """Un modèle parfait doit avoir un lift > 1 et un recall total."""
        y_true, y_prob = perfect_predictions
        metrics = compute_business_metrics(y_true, y_prob, top_k=0.30)
        assert metrics["lift"] > 1.0
        assert metrics["recall"] == 1.0

    def test_keys_present(self, perfect_predictions):
        """Toutes les clés attendues sont présentes dans la sortie."""
        y_true, y_prob = perfect_predictions
        metrics = compute_business_metrics(y_true, y_prob)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "lift" in metrics
        assert "churners_captured" in metrics
        assert "total_churners" in metrics

    def test_all_zeros(self):
        """Si personne ne part, le total de churners est 0 (pas de division par 0)."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_prob = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        metrics = compute_business_metrics(y_true, y_prob)
        assert metrics["total_churners"] == 0

    def test_recall_between_0_and_1(self, perfect_predictions):
        """Le recall doit toujours rester dans [0, 1]."""
        y_true, y_prob = perfect_predictions
        metrics = compute_business_metrics(y_true, y_prob, top_k=0.10)
        assert 0.0 <= metrics["recall"] <= 1.0


class TestPlotFunctions:
    """Tests des fonctions de visualisation (lift, ROC, calibration)."""

    @pytest.fixture
    def sample_data(self):
        """Renvoie un petit jeu de prédictions varié pour exercer les graphiques."""
        y_true = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.7, 0.3, 0.2, 0.1, 0.8, 0.4, 0.15, 0.05, 0.25])
        return y_true, y_prob

    def test_lift_curve_returns_figure(self, sample_data):
        """``plot_lift_curve`` doit renvoyer une figure matplotlib."""
        import matplotlib.pyplot as plt

        fig = plot_lift_curve(*sample_data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_roc_curve_returns_figure(self, sample_data):
        """``plot_roc_curve`` doit renvoyer une figure matplotlib."""
        import matplotlib.pyplot as plt

        fig = plot_roc_curve(*sample_data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_calibration_returns_figure(self, sample_data):
        """``plot_calibration`` doit renvoyer une figure matplotlib."""
        import matplotlib.pyplot as plt

        fig = plot_calibration(*sample_data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestFullEvaluationReport:
    """Tests du rapport d'évaluation agrégé."""

    def test_report_keys(self):
        """Toutes les clés attendues sont présentes dans le rapport agrégé."""
        y_true = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.7, 0.3, 0.2, 0.1, 0.8, 0.4, 0.15, 0.05, 0.25])
        report = full_evaluation_report(y_true, y_prob)
        assert "auc_roc" in report
        assert "avg_precision" in report
        assert "brier_score" in report
        assert "business_metrics_10pct" in report
        assert "business_metrics_15pct" in report

    def test_auc_between_0_and_1(self):
        """L'AUC doit toujours rester dans [0, 1]."""
        y_true = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.7, 0.3, 0.2, 0.1, 0.8, 0.4, 0.15, 0.05, 0.25])
        report = full_evaluation_report(y_true, y_prob)
        assert 0.0 <= report["auc_roc"] <= 1.0
