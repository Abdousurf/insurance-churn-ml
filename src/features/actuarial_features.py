"""
Actuarial Feature Engineering
================================
Domain-specific features for insurance churn prediction.
Based on actuarial pricing theory and customer behavior research.

Key insight: overpriced policies (premium >> market rate) and
negative claims experiences are the strongest churn drivers.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional


class ActuarialFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Build actuarial-inspired features for churn prediction.

    Features derived from:
    - Generalized Linear Models used in actuarial pricing
    - Customer Lifetime Value (CLV) theory
    - Behavioral economics of insurance switching
    """

    def __init__(self, market_rate_col: Optional[str] = None):
        self.market_rate_col = market_rate_col
        self.market_avg_premiums_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        # Learn market average premiums per LOB for rate comparison
        if "lob" in X.columns and "annual_premium" in X.columns:
            self.market_avg_premiums_ = (
                X.groupby("lob")["annual_premium"].median().to_dict()
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        df = self._premium_features(df)
        df = self._claims_features(df)
        df = self._loyalty_features(df)
        df = self._portfolio_features(df)
        df = self._lifecycle_features(df)

        return df

    def _premium_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Premium-related features: pricing adequacy and sensitivity."""

        if "annual_premium" in df.columns and "lob" in df.columns:
            # Premium vs. market rate (key churn driver)
            df["premium_to_market_ratio"] = df.apply(
                lambda r: r["annual_premium"] / self.market_avg_premiums_.get(r["lob"], r["annual_premium"]),
                axis=1
            )
            # Overpriced flag (>20% above market)
            df["is_overpriced"] = (df["premium_to_market_ratio"] > 1.20).astype(int)

            # Log premium (reduce skew)
            df["log_premium"] = np.log1p(df["annual_premium"])

        if "premium_change_pct" in df.columns:
            # Premium increase sensitivity
            df["premium_increased"] = (df["premium_change_pct"] > 0).astype(int)
            df["premium_increase_gt5pct"] = (df["premium_change_pct"] > 5).astype(int)

        return df

    def _claims_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Claims experience features: satisfaction proxies."""

        if "claim_count_12m" in df.columns:
            df["has_recent_claim"] = (df["claim_count_12m"] > 0).astype(int)
            df["multi_claim"] = (df["claim_count_12m"] > 1).astype(int)

        if "claim_count_all" in df.columns and "tenure_months" in df.columns:
            # Annual claims frequency
            df["claims_per_year"] = df["claim_count_all"] / (df["tenure_months"] / 12).clip(lower=0.1)

        if "claim_settled_pct" in df.columns:
            # Unresolved claims → dissatisfaction
            df["has_unsettled_claim"] = (df["claim_settled_pct"] < 1.0).astype(int)

        if "days_to_settle_avg" in df.columns:
            # Slow claims handling → churn risk
            df["slow_settlement"] = (df["days_to_settle_avg"] > 45).astype(int)

        return df

    def _loyalty_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Loyalty and retention indicators."""

        if "tenure_months" in df.columns:
            df["tenure_years"] = df["tenure_months"] / 12
            df["log_tenure"] = np.log1p(df["tenure_months"])

            # Early churn risk (< 12 months)
            df["is_new_customer"] = (df["tenure_months"] < 12).astype(int)
            # Long-term loyalty (> 5 years)
            df["is_loyal_customer"] = (df["tenure_months"] >= 60).astype(int)

        if "renewal_count" in df.columns:
            # Zero renewals = first-year customer
            df["never_renewed"] = (df["renewal_count"] == 0).astype(int)
            df["log_renewals"] = np.log1p(df["renewal_count"])

        if "last_contact_days" in df.columns:
            # Recency of last touchpoint
            df["long_since_contact"] = (df["last_contact_days"] > 180).astype(int)

        return df

    def _portfolio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-line policy indicators (bundling effect)."""

        if "policy_count_active" in df.columns:
            df["is_multi_line"] = (df["policy_count_active"] > 1).astype(int)
            # Multi-line customers have 40-60% lower churn rate (actuarial finding)
            df["multi_line_discount_eligible"] = (df["policy_count_active"] >= 2).astype(int)

        return df

    def _lifecycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Life stage and channel features."""

        if "insured_age" in df.columns:
            # Age segments (pricing-aligned)
            df["age_segment_encoded"] = pd.cut(
                df["insured_age"],
                bins=[0, 25, 35, 50, 65, 100],
                labels=[0, 1, 2, 3, 4]
            ).astype(int)

            # Young adults churn more (price-sensitive)
            df["is_young_adult"] = (df["insured_age"].between(18, 30)).astype(int)

        if "channel" in df.columns:
            # Online customers are more price-sensitive
            df["is_online_customer"] = (df["channel"] == "Online").astype(int)
            df["is_broker_customer"] = (df["channel"] == "Broker").astype(int)

        return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix for model training.
    Returns only numeric columns suitable for tree-based models.
    """
    builder = ActuarialFeatureBuilder()
    df_features = builder.fit_transform(df)

    # Select final feature set
    feature_cols = [c for c in df_features.columns if c not in [
        "policy_id", "customer_id", "inception_date", "expiry_date",
        "lob", "region", "channel",  # raw categoricals (use encoded versions)
        "churn_label",               # target
    ]]

    return df_features[feature_cols]


FEATURE_DESCRIPTIONS = {
    "premium_to_market_ratio": "Ratio of policy premium to LOB market median — key pricing adequacy signal",
    "is_overpriced": "1 if premium > 120% of market rate",
    "log_premium": "Log-transformed annual premium to reduce skewness",
    "has_recent_claim": "1 if at least one claim in last 12 months",
    "multi_claim": "1 if 2+ claims in last 12 months (experience effect)",
    "claims_per_year": "Annualized claim frequency over full tenure",
    "has_unsettled_claim": "1 if any open/pending claims (dissatisfaction proxy)",
    "slow_settlement": "1 if average claim settlement > 45 days",
    "tenure_years": "Policy duration in years",
    "is_new_customer": "1 if tenure < 12 months",
    "is_loyal_customer": "1 if tenure >= 5 years",
    "never_renewed": "1 if this is the first policy year",
    "is_multi_line": "1 if customer holds multiple active policies",
    "is_young_adult": "1 if insured age between 18-30 (high price sensitivity)",
    "is_online_customer": "1 if acquired via digital channel (higher churn risk)",
}
