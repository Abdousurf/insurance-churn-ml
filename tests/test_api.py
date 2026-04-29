"""Tests de l'API FastAPI (helpers + endpoints).

Le client de test FastAPI permet d'invoquer les endpoints en mémoire,
sans démarrer un vrai serveur HTTP. On désactive le chargement du
modèle MLflow afin que les tests tournent même sans serveur MLflow
disponible (cas typique en CI).
"""

# ───────────────────────────────────────────────────────
# RÔLE DE CE FICHIER (en clair) :
# Il vérifie que l'API répond correctement :
#   - les fonctions auxiliaires (classification du risque,
#     estimation de la valeur client, conversion DataFrame)
#     fonctionnent comme attendu,
#   - le schéma Pydantic rejette bien les requêtes invalides,
#   - les endpoints renvoient les bons codes HTTP même
#     quand le modèle n'est pas encore chargé (503).
# ───────────────────────────────────────────────────────

from contextlib import asynccontextmanager
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import (
    PolicyFeatures,
    classify_risk,
    estimate_clv,
    features_to_dataframe,
)


# ── Tests des helpers ──────────────────────────────────────────────────────


class TestClassifyRisk:
    """Tests du classement par niveau de risque."""

    def test_low_risk(self):
        """Les probabilités < 20 % sont classées ``low``."""
        assert classify_risk(0.0) == "low"
        assert classify_risk(0.1) == "low"
        assert classify_risk(0.19) == "low"

    def test_medium_risk(self):
        """Les probabilités entre 20 % et 45 % sont classées ``medium``."""
        assert classify_risk(0.2) == "medium"
        assert classify_risk(0.3) == "medium"
        assert classify_risk(0.44) == "medium"

    def test_high_risk(self):
        """Les probabilités entre 45 % et 70 % sont classées ``high``."""
        assert classify_risk(0.45) == "high"
        assert classify_risk(0.6) == "high"
        assert classify_risk(0.69) == "high"

    def test_critical_risk(self):
        """Les probabilités ≥ 70 % sont classées ``critical``."""
        assert classify_risk(0.7) == "critical"
        assert classify_risk(0.9) == "critical"
        assert classify_risk(1.0) == "critical"

    def test_boundary_values(self):
        """Les valeurs exactement aux bornes basculent dans le tier supérieur."""
        assert classify_risk(0.0) == "low"
        assert classify_risk(0.2) == "medium"
        assert classify_risk(0.45) == "high"
        assert classify_risk(0.7) == "critical"


@pytest.fixture
def sample_policy() -> PolicyFeatures:
    """Renvoie une fiche client cohérente pour les tests."""
    return PolicyFeatures(
        policy_id="TEST001",
        lob="auto",
        annual_premium=1000.0,
        tenure_months=24,
        renewal_count=2,
        claim_count_12m=0,
        claim_count_all=1,
        claim_settled_pct=1.0,
        days_to_settle_avg=20.0,
        insured_age=35,
        channel="Direct",
        policy_count_active=1,
        premium_change_pct=0.0,
    )


class TestEstimateCLV:
    """Tests de l'estimation de valeur résiduelle (CLV)."""

    def test_positive_clv(self, sample_policy):
        """La CLV doit toujours être strictement positive."""
        clv = estimate_clv(sample_policy)
        assert clv > 0

    def test_higher_premium_higher_clv(self, sample_policy):
        """Une prime plus élevée doit donner une CLV plus élevée, toutes choses égales par ailleurs."""
        clv_base = estimate_clv(sample_policy)
        sample_policy.annual_premium = 2000.0
        clv_double = estimate_clv(sample_policy)
        assert clv_double > clv_base

    def test_long_tenure_lower_clv(self):
        """Plus l'ancienneté est élevée, moins il reste d'années → CLV plus faible."""
        short = PolicyFeatures(
            policy_id="S", lob="auto", annual_premium=1000.0,
            tenure_months=6, insured_age=30, channel="Direct",
        )
        long = PolicyFeatures(
            policy_id="L", lob="auto", annual_premium=1000.0,
            tenure_months=48, insured_age=30, channel="Direct",
        )
        assert estimate_clv(short) > estimate_clv(long)


class TestFeaturesToDataframe:
    """Tests de la conversion fiche client → DataFrame pandas."""

    def test_returns_dataframe(self, sample_policy):
        """La fonction doit renvoyer un DataFrame d'une seule ligne."""
        result = features_to_dataframe(sample_policy)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_contains_expected_columns(self, sample_policy):
        """Les principales colonnes attendues sont présentes."""
        result = features_to_dataframe(sample_policy)
        assert "policy_id" in result.columns
        assert "annual_premium" in result.columns
        assert "lob" in result.columns


# ── Tests de validation Pydantic ────────────────────────────────────────────


class TestPolicyFeaturesSchema:
    """Tests de validation du schéma :class:`PolicyFeatures`."""

    def test_valid_policy(self):
        """Une fiche client valide doit être acceptée."""
        p = PolicyFeatures(
            policy_id="P1", lob="auto", annual_premium=500.0,
            tenure_months=12, insured_age=30, channel="Direct",
        )
        assert p.lob == "auto"

    def test_lob_case_insensitive(self):
        """La branche est normalisée en minuscules."""
        p = PolicyFeatures(
            policy_id="P1", lob="AUTO", annual_premium=500.0,
            tenure_months=12, insured_age=30, channel="Direct",
        )
        assert p.lob == "auto"

    def test_invalid_lob_raises(self):
        """Une branche inconnue doit lever une erreur."""
        with pytest.raises(Exception):
            PolicyFeatures(
                policy_id="P1", lob="marine", annual_premium=500.0,
                tenure_months=12, insured_age=30, channel="Direct",
            )

    def test_invalid_channel_raises(self):
        """Un canal inconnu doit lever une erreur."""
        with pytest.raises(Exception):
            PolicyFeatures(
                policy_id="P1", lob="auto", annual_premium=500.0,
                tenure_months=12, insured_age=30, channel="Phone",
            )

    def test_negative_premium_raises(self):
        """Une prime négative ou nulle doit être refusée."""
        with pytest.raises(Exception):
            PolicyFeatures(
                policy_id="P1", lob="auto", annual_premium=-100.0,
                tenure_months=12, insured_age=30, channel="Direct",
            )

    def test_age_out_of_range_raises(self):
        """Un âge hors [18, 100] doit être refusé."""
        with pytest.raises(Exception):
            PolicyFeatures(
                policy_id="P1", lob="auto", annual_premium=500.0,
                tenure_months=12, insured_age=15, channel="Direct",
            )


# ── Tests des endpoints HTTP ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Crée un ``TestClient`` avec MLflow simulé pour ne pas dépendre d'un serveur.

    On override le ``lifespan`` pour qu'il ne charge pas le modèle ; c'est
    suffisant pour tester les endpoints qui ne dépendent pas du modèle
    (``/health``, ``/model-info``) et pour vérifier les codes 503 quand
    le modèle est manquant.
    """
    with patch("src.api.main.mlflow"):
        from src.api.main import app

        @asynccontextmanager
        async def _noop_lifespan(app):
            yield

        app.router.lifespan_context = _noop_lifespan
        return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    """Tests de l'endpoint ``/health``."""

    def test_health_returns_ok(self, client):
        """``/health`` doit toujours répondre 200 avec ``status: ok``."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data


class TestModelInfoEndpoint:
    """Tests de l'endpoint ``/model-info``."""

    def test_model_info_returns_metadata(self, client):
        """``/model-info`` renvoie le nom du modèle et son stage."""
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "stage" in data


class TestPredictEndpoint:
    """Tests des endpoints de prédiction (``/predict``, ``/predict/batch``)."""

    def test_predict_without_model_returns_503(self, client):
        """Quand le modèle n'est pas chargé, ``/predict`` doit répondre 503."""
        response = client.post("/predict", json={
            "policy_id": "P1", "lob": "auto", "annual_premium": 800.0,
            "tenure_months": 24, "insured_age": 35, "channel": "Direct",
        })
        assert response.status_code == 503

    def test_predict_batch_without_model_returns_503(self, client):
        """Quand le modèle n'est pas chargé, ``/predict/batch`` doit répondre 503."""
        response = client.post("/predict/batch", json={
            "policies": [{
                "policy_id": "P1", "lob": "auto", "annual_premium": 800.0,
                "tenure_months": 24, "insured_age": 35, "channel": "Direct",
            }]
        })
        assert response.status_code == 503
