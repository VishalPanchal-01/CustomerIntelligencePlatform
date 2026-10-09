from typing import Any, Optional

from pydantic import (
    BaseModel,
    Field
)


# ============================================================
# HEALTH RESPONSE
# ============================================================

class HealthResponse(BaseModel):

    status: str

    service: str

    version: str


# ============================================================
# API INFORMATION
# ============================================================

class APIInfoResponse(BaseModel):

    name: str

    version: str

    description: str

    endpoints: list[str]


# ============================================================
# DATASET STATUS
# ============================================================

class DatasetStatusResponse(BaseModel):

    available: bool

    rows: int = 0

    customers: int = 0

    columns: int = 0

    path: str


# ============================================================
# CUSTOMER SUMMARY
# ============================================================

class CustomerSummaryResponse(BaseModel):

    customer_id: str

    segment: Optional[str] = None

    churn_probability: Optional[float] = None

    churn_risk: Optional[str] = None

    predicted_90_day_revenue: Optional[float] = None

    clv_value_band: Optional[str] = None

    top_recommended_product: Optional[str] = None

    recommendation_source: Optional[str] = None


# ============================================================
# COMPLETE CUSTOMER
# ============================================================

class CustomerIntelligenceResponse(BaseModel):

    customer_id: str

    intelligence: dict[str, Any]


# ============================================================
# CUSTOMER LIST
# ============================================================

class CustomerListItem(BaseModel):

    customer_id: str

    segment: Optional[str] = None

    churn_risk: Optional[str] = None

    clv_value_band: Optional[str] = None


class CustomerListResponse(BaseModel):

    total: int

    limit: int

    offset: int

    customers: list[CustomerListItem]


# ============================================================
# CHURN
# ============================================================

class ChurnPredictionRequest(BaseModel):

    Recency: float = Field(ge=0)
    Frequency: float = Field(ge=0)
    Monetary: float = Field(ge=0)
    TotalItems: float = Field(ge=0)
    AverageOrderValue: float = Field(ge=0)
    Tenure: float = Field(ge=0)


class ChurnPredictionResponse(BaseModel):

    churn_probability: float

    non_churn_probability: float

    predicted_class: int

    prediction_label: str

    churn_risk: str

    decision_threshold: float

    features: dict[str, float]


class ChurnModelStatusResponse(BaseModel):

    available: bool

    loaded: bool

    model_path: str

    feature_count: int

    features: list[str]

    decision_threshold: float

    low_risk_threshold: float

    high_risk_threshold: float


# ============================================================
# CLV
# ============================================================

class CLVPredictionRequest(BaseModel):

    Recency: float = Field(ge=0)
    Frequency: float = Field(ge=0)
    Monetary: float = Field(ge=0)
    TotalItems: float = Field(ge=0)
    AverageOrderValue: float = Field(ge=0)
    Tenure: float = Field(ge=0)


class CLVPredictionResponse(BaseModel):

    predicted_90_day_revenue: float

    clv_value_band: str

    lower_value_threshold: float

    upper_value_threshold: float

    features: dict[str, float]


class CLVModelStatusResponse(BaseModel):

    model_available: bool

    model_loaded: bool

    value_bands_available: bool

    value_bands_loaded: bool

    model_path: str

    value_bands_path: str

    feature_count: int

    features: list[str]


# ============================================================
# RECOMMENDATION
# ============================================================

class RecommendationItem(BaseModel):

    rank: int

    stock_code: Optional[str] = None

    product: Optional[str] = None

    score: Optional[float] = None


class RecommendationResponse(BaseModel):

    customer_id: str

    recommendation_source: str

    recommendation_mode: str

    top_n: int

    cold_start: bool

    recommendations: list[RecommendationItem]


class RecommendationModelStatusResponse(BaseModel):

    model_available: bool

    model_loaded: bool

    batch_predictions_available: bool

    batch_predictions_loaded: bool

    model_path: str

    batch_predictions_path: str

    known_customers: int

    cold_start_supported: bool


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

class SHAPFeatureContribution(BaseModel):

    feature: str

    feature_value: float

    shap_value: float

    absolute_shap: float

    impact_direction: str

    impact_rank: Optional[int] = None


class CustomerExplanationResponse(BaseModel):

    customer_id: str

    model: str

    explanation_available: bool

    top_positive_drivers: list[SHAPFeatureContribution]

    top_negative_drivers: list[SHAPFeatureContribution]

    all_contributions: list[SHAPFeatureContribution]

    interpretation_note: str


class GlobalSHAPFeature(BaseModel):

    feature: str

    mean_absolute_shap: float

    importance_rank: Optional[int] = None


class GlobalExplanationResponse(BaseModel):

    model: str

    available: bool

    feature_importance: list[GlobalSHAPFeature]

    interpretation_note: str


class ExplainabilityStatusResponse(BaseModel):

    churn_global_available: bool

    churn_local_available: bool

    clv_global_available: bool

    clv_local_available: bool

    churn_explained_customers: int

    clv_explained_customers: int


# ============================================================
# BUSINESS DECISION
# ============================================================

class BusinessDecisionResponse(BaseModel):

    priority: str

    recommended_action: str


# ============================================================
# UNIFIED INTELLIGENCE
# ============================================================

class UnifiedCustomerIntelligenceResponse(BaseModel):

    customer_id: str

    profile: dict[str, Any]

    churn: dict[str, Any]

    clv: dict[str, Any]

    recommendation: Optional[RecommendationResponse] = None

    churn_explanation: CustomerExplanationResponse

    clv_explanation: CustomerExplanationResponse

    business_decision: BusinessDecisionResponse


# ============================================================
# ERROR
# ============================================================

class ErrorResponse(BaseModel):

    detail: str