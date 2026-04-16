"""Tests for the insurance feature building code.

Makes sure the feature builder correctly creates all expected columns,
flags the right customers, and handles a wide range of inputs without errors.
"""

# ───────────────────────────────────────────────────────
# WHAT THIS FILE DOES (in plain English):
# This file tests the feature engineering code to make
# sure it works correctly. It checks things like:
# - Are all expected columns created?
# - Are customers correctly flagged as "overpriced",
#   "new", "loyal", etc.?
# - Does the code handle unusual inputs without crashing?
# ───────────────────────────────────────────────────────

import pytest
import pandas as pd
from hypothesis import given, strategies as st, settings

from src.features.actuarial_features import ActuarialFeatureBuilder, build_feature_matrix


@pytest.fixture
def sample_data():
    """Create a small fake dataset of insurance policies for testing.

    Returns:
        A table with 5 made-up policies covering different customer scenarios.
    """
    return pd.DataFrame({
        "policy_id": ["POL001", "POL002", "POL003", "POL004", "POL005"],
        "lob": ["auto", "home", "auto", "health", "auto"],
        "annual_premium": [800.0, 520.0, 1200.0, 600.0, 900.0],
        "tenure_months": [24, 60, 6, 36, 120],
        "renewal_count": [2, 5, 0, 3, 10],
        "claim_count_12m": [0, 1, 2, 0, 0],
        "claim_count_all": [1, 3, 2, 0, 5],
        "claim_settled_pct": [1.0, 0.8, 0.5, 1.0, 1.0],
        "days_to_settle_avg": [20.0, 50.0, 60.0, 0.0, 15.0],
        "insured_age": [35, 55, 22, 40, 68],
        "channel": ["Direct", "Broker", "Online", "Agent", "Direct"],
        "policy_count_active": [1, 3, 1, 2, 2],
        "premium_change_pct": [0.0, 3.0, 12.0, -2.0, 1.5],
        "last_contact_days": [30, 200, 10, None, 365],
    })


@pytest.fixture
def builder():
    """Create a fresh feature builder for testing.

    Returns:
        A new, unused feature builder.
    """
    return ActuarialFeatureBuilder()


class TestActuarialFeatureBuilder:
    """Tests for the feature builder that creates insurance-specific columns."""

    def test_fit_learns_market_premiums(self, builder, sample_data):
        # After fitting, the builder should know the typical price for each insurance type
        builder.fit(sample_data)
        assert "auto" in builder.market_avg_premiums_
        assert "home" in builder.market_avg_premiums_
        assert builder.market_avg_premiums_["auto"] > 0

    def test_transform_adds_premium_features(self, builder, sample_data):
        # Check that pricing-related columns get created
        builder.fit(sample_data)
        result = builder.transform(sample_data)
        assert "premium_to_market_ratio" in result.columns
        assert "is_overpriced" in result.columns
        assert "log_premium" in result.columns

    def test_transform_adds_claims_features(self, builder, sample_data):
        # Check that claims-related columns get created
        builder.fit(sample_data)
        result = builder.transform(sample_data)
        assert "has_recent_claim" in result.columns
        assert "multi_claim" in result.columns
        assert "claims_per_year" in result.columns
        assert "has_unsettled_claim" in result.columns
        assert "slow_settlement" in result.columns

    def test_transform_adds_loyalty_features(self, builder, sample_data):
        # Check that loyalty-related columns get created
        builder.fit(sample_data)
        result = builder.transform(sample_data)
        assert "tenure_years" in result.columns
        assert "is_new_customer" in result.columns
        assert "is_loyal_customer" in result.columns
        assert "never_renewed" in result.columns

    def test_transform_adds_portfolio_features(self, builder, sample_data):
        # Check that multi-policy columns get created
        builder.fit(sample_data)
        result = builder.transform(sample_data)
        assert "is_multi_line" in result.columns

    def test_transform_adds_lifecycle_features(self, builder, sample_data):
        # Check that age and channel columns get created
        builder.fit(sample_data)
        result = builder.transform(sample_data)
        assert "age_segment_encoded" in result.columns
        assert "is_young_adult" in result.columns
        assert "is_online_customer" in result.columns
        assert "is_broker_customer" in result.columns

    def test_overpriced_flag(self, builder, sample_data):
        # POL003 pays 1200 for auto, but the typical auto price is 900 — that's 33% above market
        builder.fit(sample_data)
        result = builder.transform(sample_data)
        pol3 = result[sample_data["policy_id"] == "POL003"]
        assert pol3["is_overpriced"].values[0] == 1

    def test_new_customer_flag(self, builder, sample_data):
        # POL003 has only been a customer for 6 months — should be flagged as new
        builder.fit(sample_data)
        result = builder.transform(sample_data)
        pol3 = result[sample_data["policy_id"] == "POL003"]
        assert pol3["is_new_customer"].values[0] == 1

    def test_loyal_customer_flag(self, builder, sample_data):
        # POL005 has been a customer for 120 months (10 years) — should be flagged as loyal
        builder.fit(sample_data)
        result = builder.transform(sample_data)
        pol5 = result[sample_data["policy_id"] == "POL005"]
        assert pol5["is_loyal_customer"].values[0] == 1

    def test_multi_line_flag(self, builder, sample_data):
        # POL002 has 3 active policies — should be flagged as multi-line
        builder.fit(sample_data)
        result = builder.transform(sample_data)
        pol2 = result[sample_data["policy_id"] == "POL002"]
        assert pol2["is_multi_line"].values[0] == 1

    def test_slow_settlement_flag(self, builder, sample_data):
        # POL003 has average settlement time of 60 days (above the 45-day threshold)
        builder.fit(sample_data)
        result = builder.transform(sample_data)
        pol3 = result[sample_data["policy_id"] == "POL003"]
        assert pol3["slow_settlement"].values[0] == 1

    def test_fit_transform_equals_fit_then_transform(self, builder, sample_data):
        # Doing fit_transform in one step should give the same result as doing them separately
        result_ft = builder.fit_transform(sample_data)
        builder2 = ActuarialFeatureBuilder()
        builder2.fit(sample_data)
        result_sep = builder2.transform(sample_data)
        pd.testing.assert_frame_equal(result_ft, result_sep)

    def test_transform_preserves_row_count(self, builder, sample_data):
        # The number of rows should stay the same — we're adding columns, not removing rows
        builder.fit(sample_data)
        result = builder.transform(sample_data)
        assert len(result) == len(sample_data)

    def test_no_nans_in_binary_features(self, builder, sample_data):
        # All yes/no flag columns should only contain 0 or 1, never missing values
        builder.fit(sample_data)
        result = builder.transform(sample_data)
        binary_cols = [
            "is_overpriced", "has_recent_claim", "multi_claim",
            "is_new_customer", "is_loyal_customer", "never_renewed",
            "is_multi_line", "is_young_adult", "is_online_customer",
        ]
        for col in binary_cols:
            assert result[col].isna().sum() == 0, f"NaNs found in {col}"
            assert set(result[col].unique()).issubset({0, 1}), f"{col} has non-binary values"


class TestBuildFeatureMatrix:
    """Tests for the helper function that builds the complete feature set."""

    def test_excludes_non_feature_cols(self, sample_data):
        # The output should NOT contain IDs, text fields, or the answer column
        sample_data["churn_label"] = [0, 1, 0, 1, 0]
        result = build_feature_matrix(sample_data)
        assert "policy_id" not in result.columns
        assert "churn_label" not in result.columns
        assert "lob" not in result.columns
        assert "channel" not in result.columns

    def test_output_is_numeric(self, sample_data):
        # Every column in the output should be a number (not text)
        sample_data["churn_label"] = [0, 1, 0, 1, 0]
        result = build_feature_matrix(sample_data)
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} is not numeric"


# This test uses random data generation to make sure the feature builder
# doesn't crash on any valid combination of inputs
@given(
    premium=st.floats(min_value=100, max_value=50000, allow_nan=False),
    tenure=st.integers(min_value=1, max_value=600),
    age=st.integers(min_value=18, max_value=100),
    claim_count=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=50)
def test_feature_builder_no_crash_on_valid_input(premium, tenure, age, claim_count):
    """Make sure the feature builder works with any reasonable input values.

    This test automatically generates 50 random combinations of customer
    data and checks that the builder handles them all without errors.

    Args:
        premium: A randomly chosen annual premium amount.
        tenure: A randomly chosen number of months as a customer.
        age: A randomly chosen age for the insured person.
        claim_count: A randomly chosen number of claims.
    """
    # Create a single-row dataset with the random values
    df = pd.DataFrame({
        "lob": ["auto"],
        "annual_premium": [premium],
        "tenure_months": [tenure],
        "renewal_count": [0],
        "claim_count_12m": [claim_count],
        "claim_count_all": [claim_count],
        "claim_settled_pct": [1.0],
        "days_to_settle_avg": [30.0],
        "insured_age": [age],
        "channel": ["Direct"],
        "policy_count_active": [1],
        "premium_change_pct": [0.0],
    })
    # Run the feature builder — it should work without errors
    builder = ActuarialFeatureBuilder()
    result = builder.fit_transform(df)
    assert len(result) == 1
    assert result["log_premium"].values[0] > 0
