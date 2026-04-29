"""Pure utility functions for insurance churn predictions.

These functions have no dependency on MLflow or the FastAPI app,
so they can be imported safely in tests and other modules.
"""

import pandas as pd

from src.api.schemas import PolicyFeatures

# Risk tiers define the probability ranges for each category
RISK_TIERS = {
    "low": (0.0, 0.2),
    "medium": (0.2, 0.45),
    "high": (0.45, 0.70),
    "critical": (0.70, 1.0),
}

# What action to take for each risk level
RETENTION_ACTIONS = {
    "low": "no_action",
    "medium": "soft_retention",   # loyalty email, mild offer
    "high": "proactive_outreach",  # call, 5-10% discount offer
    "critical": "urgent_retention", # personal advisor, max discount
}


def classify_risk(prob: float) -> str:
    """Turn a churn probability number into a risk category name.

    Args:
        prob: The predicted chance of churning, between 0 and 1.

    Returns:
        A risk level: "low", "medium", "high", or "critical".
    """
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= prob < hi:
            return tier
    return "critical"


def estimate_clv(features: PolicyFeatures) -> float:
    """Estimate how much revenue this customer is worth over their remaining lifetime.

    Uses a simplified financial formula based on how much they pay per year
    and how many years they're likely to stay.

    Args:
        features: The customer's policy details, including premium and tenure.

    Returns:
        The estimated lifetime value in EUR.
    """
    # Estimate how many more years this customer might stay (up to 5 total)
    expected_tenure_years = max(1, (60 - features.tenure_months) / 12)
    discount_rate = 0.08
    # Standard discounted cash flow calculation
    clv = features.annual_premium * (1 - (1 + discount_rate) ** -expected_tenure_years) / discount_rate
    return round(clv, 2)


def features_to_dataframe(policy: PolicyFeatures) -> pd.DataFrame:
    """Convert a single customer's data into the table format the model expects.

    Args:
        policy: The customer's policy details.

    Returns:
        A one-row data table ready for the model.
    """
    return pd.DataFrame([policy.model_dump()])
