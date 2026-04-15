"""Create insurance-specific data columns that help predict customer churn.

These features are based on how insurance companies price policies and what
research shows about why customers leave. The main idea: customers who are
paying too much compared to the market, or who had bad claims experiences,
are the most likely to leave.
"""

# ───────────────────────────────────────────────────────
# WHAT THIS FILE DOES (in plain English):
# This file takes raw insurance policy data and creates
# new calculated columns (features) that capture important
# patterns — like whether a customer is overpaying, how
# often they file claims, how long they've been a customer,
# and other signals that help predict if they'll leave.
# ───────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional


class ActuarialFeatureBuilder(BaseEstimator, TransformerMixin):
    """A tool that creates insurance-specific data columns for churn prediction.

    It looks at things like pricing, claims history, customer loyalty,
    how many policies someone holds, and their age/channel to build
    useful signals for the model.

    Attributes:
        market_rate_col: Optional name of a column with external market prices.
        market_avg_premiums_: A lookup of typical premiums for each type of insurance,
            learned from the data.
    """

    def __init__(self, market_rate_col: Optional[str] = None):
        """Set up the feature builder.

        Args:
            market_rate_col: Optional column name with external market price data.
                If not provided, market prices are estimated from the data itself.
        """
        self.market_rate_col = market_rate_col
        self.market_avg_premiums_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Learn the typical (median) premium for each type of insurance.

        This lets us later compare each customer's premium to the market
        average and see if they're overpaying.

        Args:
            X: Table of policy data to learn from.
            y: Not used. Only here so this works with standard machine learning tools.

        Returns:
            self
        """
        # Calculate the middle-of-the-road premium for each insurance type
        if "lob" in X.columns and "annual_premium" in X.columns:
            self.market_avg_premiums_ = (
                X.groupby("lob")["annual_premium"].median().to_dict()
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add all the calculated insurance columns to the data.

        Args:
            X: Table of policy data to add features to.

        Returns:
            The same table but with new calculated columns added.
        """
        df = X.copy()

        # Add each group of features one at a time
        df = self._premium_features(df)
        df = self._claims_features(df)
        df = self._loyalty_features(df)
        df = self._portfolio_features(df)
        df = self._lifecycle_features(df)

        return df

    def _premium_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create columns related to pricing — are customers paying a fair price?

        Args:
            df: Table of policy data.

        Returns:
            The same table with pricing-related columns added.
        """

        if "annual_premium" in df.columns and "lob" in df.columns:
            # How does each customer's price compare to the typical market price?
            df["premium_to_market_ratio"] = df.apply(
                lambda r: r["annual_premium"] / self.market_avg_premiums_.get(r["lob"], r["annual_premium"]),
                axis=1
            )
            # Flag customers paying more than 20% above market rate
            df["is_overpriced"] = (df["premium_to_market_ratio"] > 1.20).astype(int)

            # Shrink very large premium values so they don't dominate the model
            df["log_premium"] = np.log1p(df["annual_premium"])

        if "premium_change_pct" in df.columns:
            # Did their premium go up this year?
            df["premium_increased"] = (df["premium_change_pct"] > 0).astype(int)
            # Did it go up by more than 5%? (big increases push people to leave)
            df["premium_increase_gt5pct"] = (df["premium_change_pct"] > 5).astype(int)

        return df

    def _claims_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create columns about claims history — a rough measure of customer satisfaction.

        Customers who filed claims recently, had multiple claims, or waited
        a long time for settlement are more likely to be unhappy and leave.

        Args:
            df: Table of policy data.

        Returns:
            The same table with claims-related columns added.
        """

        if "claim_count_12m" in df.columns:
            # Did they file any claims in the last year?
            df["has_recent_claim"] = (df["claim_count_12m"] > 0).astype(int)
            # Did they file more than one claim recently?
            df["multi_claim"] = (df["claim_count_12m"] > 1).astype(int)

        if "claim_count_all" in df.columns and "tenure_months" in df.columns:
            # On average, how many claims do they file per year?
            df["claims_per_year"] = df["claim_count_all"] / (df["tenure_months"] / 12).clip(lower=0.1)

        if "claim_settled_pct" in df.columns:
            # Do they have any unresolved claims? (this frustrates customers)
            df["has_unsettled_claim"] = (df["claim_settled_pct"] < 1.0).astype(int)

        if "days_to_settle_avg" in df.columns:
            # Were their claims slow to process? (over 45 days is considered slow)
            df["slow_settlement"] = (df["days_to_settle_avg"] > 45).astype(int)

        return df

    def _loyalty_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create columns about how long and loyal a customer has been.

        New customers and those who haven't been contacted recently are
        at higher risk of leaving.

        Args:
            df: Table of policy data.

        Returns:
            The same table with loyalty-related columns added.
        """

        if "tenure_months" in df.columns:
            # Convert months to years for easier reading
            df["tenure_years"] = df["tenure_months"] / 12
            # Shrink very long tenures so they don't dominate
            df["log_tenure"] = np.log1p(df["tenure_months"])

            # Brand new customers (less than 1 year) — high churn risk
            df["is_new_customer"] = (df["tenure_months"] < 12).astype(int)
            # Long-term customers (5+ years) — usually very loyal
            df["is_loyal_customer"] = (df["tenure_months"] >= 60).astype(int)

        if "renewal_count" in df.columns:
            # Never renewed = first-year customer, higher risk
            df["never_renewed"] = (df["renewal_count"] == 0).astype(int)
            # Shrink large renewal counts
            df["log_renewals"] = np.log1p(df["renewal_count"])

        if "last_contact_days" in df.columns:
            # Haven't heard from us in over 6 months — might feel forgotten
            df["long_since_contact"] = (df["last_contact_days"] > 180).astype(int)

        return df

    def _portfolio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create columns about whether customers hold multiple policies.

        Customers with several policies (e.g., auto + home) are much less
        likely to leave — they're more "locked in" and often get bundle discounts.

        Args:
            df: Table of policy data.

        Returns:
            The same table with multi-policy columns added.
        """

        if "policy_count_active" in df.columns:
            # Does this customer have more than one policy with us?
            df["is_multi_line"] = (df["policy_count_active"] > 1).astype(int)
            # Customers with 2+ policies could qualify for a bundle discount
            df["multi_line_discount_eligible"] = (df["policy_count_active"] >= 2).astype(int)

        return df

    def _lifecycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create columns about the customer's age group and how they signed up.

        Younger customers and those who signed up online tend to shop around
        more and are more sensitive to price.

        Args:
            df: Table of policy data.

        Returns:
            The same table with age and channel columns added.
        """

        if "insured_age" in df.columns:
            # Group ages into brackets that match insurance pricing tiers
            df["age_segment_encoded"] = pd.cut(
                df["insured_age"],
                bins=[0, 25, 35, 50, 65, 100],
                labels=[0, 1, 2, 3, 4]
            ).astype(int)

            # Young adults (18-30) are the most price-sensitive age group
            df["is_young_adult"] = (df["insured_age"].between(18, 30)).astype(int)

        if "channel" in df.columns:
            # Online customers tend to compare prices more and switch more easily
            df["is_online_customer"] = (df["channel"] == "Online").astype(int)
            # Broker customers often have a personal relationship keeping them loyal
            df["is_broker_customer"] = (df["channel"] == "Broker").astype(int)

        return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create the complete set of features ready for model training.

    Takes raw policy data, adds all the insurance-specific calculated columns,
    then removes any columns the model shouldn't see (like IDs, text fields,
    and the answer column).

    Args:
        df: Raw policy data table including all needed columns.

    Returns:
        A table with only the numeric feature columns the model can use.
    """
    # Create the builder and generate all features
    builder = ActuarialFeatureBuilder()
    df_features = builder.fit_transform(df)

    # Remove columns that aren't useful for the model
    feature_cols = [c for c in df_features.columns if c not in [
        "policy_id", "customer_id", "inception_date", "expiry_date",
        "lob", "region", "channel",  # raw text fields (we use encoded versions instead)
        "churn_label",               # the answer — model shouldn't peek at this
    ]]

    return df_features[feature_cols]


# A human-readable description of each feature we create, useful for reports
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
