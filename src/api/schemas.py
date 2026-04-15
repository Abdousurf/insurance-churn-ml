"""Data shapes for the churn prediction API — defines what goes in and what comes out.

These schemas make sure that incoming requests have all the right fields
and that responses always follow a consistent format.
"""

# ───────────────────────────────────────────────────────
# WHAT THIS FILE DOES (in plain English):
# This file defines the "shapes" of data that the API
# accepts and returns. It makes sure that:
# - Incoming customer data has all required fields
# - Field values are within valid ranges (e.g., age 18-100)
# - Response data always follows the same structure
# Think of it as a contract between the API and its users.
# ───────────────────────────────────────────────────────

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PolicyFeatures(BaseModel):
    """The customer data needed to make a churn prediction.

    Attributes:
        policy_id: A unique identifier for the policy.
        lob: Type of insurance (auto, home, liability, health).
        annual_premium: How much the customer pays per year in EUR.
        tenure_months: How many months this customer has been with us.
        renewal_count: How many times they've renewed their policy.
        claim_count_12m: How many claims they've filed in the last year.
        claim_count_all: Total claims they've ever filed with us.
        claim_settled_pct: What fraction of their claims have been resolved (0 to 1).
        days_to_settle_avg: How many days it takes on average to settle their claims.
        insured_age: How old the insured person is.
        channel: How they signed up (Direct, Broker, Online, Agent).
        policy_count_active: How many active policies they currently hold.
        premium_change_pct: How much their premium changed from last year (in percent).
        last_contact_days: How many days since we last contacted them, if known.
    """

    policy_id: str
    lob: str = Field(..., description="Line of Business: auto, home, liability, health")
    annual_premium: float = Field(..., gt=0, description="Annual premium in EUR")
    tenure_months: int = Field(..., ge=0, description="Policy tenure in months")
    renewal_count: int = Field(0, ge=0)
    claim_count_12m: int = Field(0, ge=0)
    claim_count_all: int = Field(0, ge=0)
    claim_settled_pct: float = Field(1.0, ge=0, le=1)
    days_to_settle_avg: float = Field(30.0, ge=0)
    insured_age: int = Field(..., ge=18, le=100)
    channel: str = Field(..., description="Acquisition channel: Direct, Broker, Online, Agent")
    policy_count_active: int = Field(1, ge=1)
    premium_change_pct: float = Field(0.0)
    last_contact_days: Optional[int] = None

    @field_validator("lob")
    @classmethod
    def validate_lob(cls, v):
        """Check that the insurance type is one we support and make it lowercase.

        Args:
            v: The insurance type value provided by the user.

        Returns:
            The insurance type in lowercase.

        Raises:
            ValueError: If the insurance type isn't one of: auto, home, liability, health.
        """
        allowed = {"auto", "home", "liability", "health"}
        if v.lower() not in allowed:
            raise ValueError(f"lob must be one of {allowed}")
        return v.lower()

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v):
        """Check that the sign-up channel is one we recognize.

        Args:
            v: The channel value provided by the user.

        Returns:
            The channel value, unchanged if valid.

        Raises:
            ValueError: If the channel isn't one of: Direct, Broker, Online, Agent.
        """
        allowed = {"Direct", "Broker", "Online", "Agent"}
        if v not in allowed:
            raise ValueError(f"channel must be one of {allowed}")
        return v


class ChurnPrediction(BaseModel):
    """The prediction result for a single customer.

    Attributes:
        policy_id: Which policy was scored.
        churn_probability: How likely the customer is to leave (0 to 1).
        risk_tier: A simple category: low, medium, high, or critical.
        recommended_action: What the retention team should do about this customer.
        estimated_clv: How much revenue this customer is expected to bring over time, in EUR.
        model_version: Which version of the model made this prediction.
    """

    policy_id: str
    churn_probability: float
    risk_tier: str
    recommended_action: str
    estimated_clv: float
    model_version: str


class BatchRequest(BaseModel):
    """A request to predict churn for multiple customers at once.

    Attributes:
        policies: A list of customer records to score.
    """

    policies: list[PolicyFeatures]


class BatchResponse(BaseModel):
    """The results of a batch prediction, with a summary of risk.

    Attributes:
        predictions: Individual prediction results for each customer.
        total_at_risk: How many customers are in the high or critical risk groups.
        total_premium_at_risk: Total yearly premium from those high-risk customers, in EUR.
    """

    predictions: list[ChurnPrediction]
    total_at_risk: int
    total_premium_at_risk: float
