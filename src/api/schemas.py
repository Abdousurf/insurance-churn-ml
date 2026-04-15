"""Pydantic schemas for the insurance churn prediction API.

Defines request and response models for single and batch prediction
endpoints, including input validation for policy features.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PolicyFeatures(BaseModel):
    """Input schema for a single policyholder's features.

    Attributes:
        policy_id: Unique policy identifier.
        lob: Line of business (auto, home, liability, health).
        annual_premium: Annual premium amount in EUR.
        tenure_months: Policy tenure in months.
        renewal_count: Number of past renewals.
        claim_count_12m: Claims filed in the last 12 months.
        claim_count_all: Total claims filed over the policy lifetime.
        claim_settled_pct: Fraction of claims that have been settled.
        days_to_settle_avg: Average days to settle a claim.
        insured_age: Age of the insured person.
        channel: Acquisition channel (Direct, Broker, Online, Agent).
        policy_count_active: Number of active policies held by the customer.
        premium_change_pct: Year-over-year premium change percentage.
        last_contact_days: Days since last customer contact, if known.
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
        """Validate and normalize the line of business field.

        Args:
            v: Raw lob string value.

        Returns:
            Lowercased lob string.

        Raises:
            ValueError: If lob is not one of the allowed values.
        """
        allowed = {"auto", "home", "liability", "health"}
        if v.lower() not in allowed:
            raise ValueError(f"lob must be one of {allowed}")
        return v.lower()

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v):
        """Validate the acquisition channel field.

        Args:
            v: Raw channel string value.

        Returns:
            The validated channel string.

        Raises:
            ValueError: If channel is not one of the allowed values.
        """
        allowed = {"Direct", "Broker", "Online", "Agent"}
        if v not in allowed:
            raise ValueError(f"channel must be one of {allowed}")
        return v


class ChurnPrediction(BaseModel):
    """Response schema for a single churn prediction.

    Attributes:
        policy_id: The policy that was scored.
        churn_probability: Predicted probability of churn (0 to 1).
        risk_tier: Categorical risk level (low, medium, high, critical).
        recommended_action: Suggested retention action for this tier.
        estimated_clv: Estimated customer lifetime value in EUR.
        model_version: Identifier of the model version used.
    """

    policy_id: str
    churn_probability: float
    risk_tier: str
    recommended_action: str
    estimated_clv: float
    model_version: str


class BatchRequest(BaseModel):
    """Request schema for batch churn predictions.

    Attributes:
        policies: List of PolicyFeatures to score.
    """

    policies: list[PolicyFeatures]


class BatchResponse(BaseModel):
    """Response schema for batch churn predictions.

    Attributes:
        predictions: List of individual ChurnPrediction results.
        total_at_risk: Count of policies in high or critical risk tiers.
        total_premium_at_risk: Sum of annual premiums for at-risk policies.
    """

    predictions: list[ChurnPrediction]
    total_at_risk: int
    total_premium_at_risk: float
