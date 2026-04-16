"""Tests for the FastAPI churn prediction API.

Tests the helper functions (risk classification, CLV estimation)
and the API endpoints using FastAPI's TestClient.
"""

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from src.api.main import classify_risk, estimate_clv, features_to_dataframe, app
from src.api.schemas import PolicyFeatures


# ── Helper function tests ──────────────────────────────────────────────────


class TestClassifyRisk:
    """Tests for the risk tier classification function."""

    def test_low_risk(self):
        assert classify_risk(0.0) == "low"
        assert classify_risk(0.1) == "low"
        assert classify_risk(0.19) == "low"

    def test_medium_risk(self):
        assert classify_risk(0.2) == "medium"
        assert classify_risk(0.3) == "medium"
        assert classify_risk(0.44) == "medium"

    def test_high_risk(self):
        assert classify_risk(0.45) == "high"
        assert classify_risk(0.6) == "high"
        assert classify_risk(0.69) == "high"

    def test_critical_risk(self):
        assert classify_risk(0.7) == "critical"
        assert classify_risk(0.9) == "critical"
        assert classify_risk(1.0) == "critical"

    def test_boundary_values(self):
        assert classify_risk(0.0) == "low"
        assert classify_risk(0.2) == "medium"
        assert classify_risk(0.45) == "high"
        assert classify_risk(0.7) == "critical"


@pytest.fixture
def sample_policy():
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
    """Tests for the customer lifetime value estimation."""

    def test_positive_clv(self, sample_policy):
        clv = estimate_clv(sample_policy)
        assert clv > 0

    def test_higher_premium_higher_clv(self, sample_policy):
        clv_base = estimate_clv(sample_policy)
        sample_policy.annual_premium = 2000.0
        clv_double = estimate_clv(sample_policy)
        assert clv_double > clv_base

    def test_long_tenure_lower_clv(self):
        """Longer tenure -> less remaining years -> lower CLV."""
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
    """Tests for converting policy schemas to DataFrames."""

    def test_returns_dataframe(self, sample_policy):
        result = features_to_dataframe(sample_policy)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_contains_expected_columns(self, sample_policy):
        result = features_to_dataframe(sample_policy)
        assert "policy_id" in result.columns
        assert "annual_premium" in result.columns
        assert "lob" in result.columns


# ── Pydantic schema validation tests ──────────────────────────────────────


class TestPolicyFeaturesSchema:
    """Tests for the PolicyFeatures Pydantic model validation."""

    def test_valid_policy(self):
        p = PolicyFeatures(
            policy_id="P1", lob="auto", annual_premium=500.0,
            tenure_months=12, insured_age=30, channel="Direct",
        )
        assert p.lob == "auto"

    def test_lob_case_insensitive(self):
        p = PolicyFeatures(
            policy_id="P1", lob="AUTO", annual_premium=500.0,
            tenure_months=12, insured_age=30, channel="Direct",
        )
        assert p.lob == "auto"

    def test_invalid_lob_raises(self):
        with pytest.raises(Exception):
            PolicyFeatures(
                policy_id="P1", lob="marine", annual_premium=500.0,
                tenure_months=12, insured_age=30, channel="Direct",
            )

    def test_invalid_channel_raises(self):
        with pytest.raises(Exception):
            PolicyFeatures(
                policy_id="P1", lob="auto", annual_premium=500.0,
                tenure_months=12, insured_age=30, channel="Phone",
            )

    def test_negative_premium_raises(self):
        with pytest.raises(Exception):
            PolicyFeatures(
                policy_id="P1", lob="auto", annual_premium=-100.0,
                tenure_months=12, insured_age=30, channel="Direct",
            )

    def test_age_out_of_range_raises(self):
        with pytest.raises(Exception):
            PolicyFeatures(
                policy_id="P1", lob="auto", annual_premium=500.0,
                tenure_months=12, insured_age=15, channel="Direct",
            )


# ── API endpoint tests ────────────────────────────────────────────────────

client = TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data


class TestModelInfoEndpoint:
    """Tests for the /model-info endpoint."""

    def test_model_info_returns_metadata(self):
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "stage" in data


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_without_model_returns_503(self):
        response = client.post("/predict", json={
            "policy_id": "P1", "lob": "auto", "annual_premium": 800.0,
            "tenure_months": 24, "insured_age": 35, "channel": "Direct",
        })
        assert response.status_code == 503

    def test_predict_batch_without_model_returns_503(self):
        response = client.post("/predict/batch", json={
            "policies": [{
                "policy_id": "P1", "lob": "auto", "annual_premium": 800.0,
                "tenure_months": 24, "insured_age": 35, "channel": "Direct",
            }]
        })
        assert response.status_code == 503
