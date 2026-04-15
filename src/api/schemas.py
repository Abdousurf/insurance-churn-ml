from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PolicyFeatures(BaseModel):
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
        allowed = {"auto", "home", "liability", "health"}
        if v.lower() not in allowed:
            raise ValueError(f"lob must be one of {allowed}")
        return v.lower()

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v):
        allowed = {"Direct", "Broker", "Online", "Agent"}
        if v not in allowed:
            raise ValueError(f"channel must be one of {allowed}")
        return v


class ChurnPrediction(BaseModel):
    policy_id: str
    churn_probability: float
    risk_tier: str
    recommended_action: str
    estimated_clv: float
    model_version: str


class BatchRequest(BaseModel):
    policies: list[PolicyFeatures]


class BatchResponse(BaseModel):
    predictions: list[ChurnPrediction]
    total_at_risk: int
    total_premium_at_risk: float
