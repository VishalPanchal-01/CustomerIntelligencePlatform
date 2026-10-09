import os
import sys

from fastapi import (
    FastAPI,
    HTTPException,
    Query
)

from fastapi.middleware.cors import (
    CORSMiddleware
)


# ============================================================
# PROJECT ROOT
# ============================================================

CURRENT_DIRECTORY = os.path.dirname(
    os.path.abspath(
        __file__
    )
)

PROJECT_ROOT = os.path.dirname(
    CURRENT_DIRECTORY
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# IMPORTS
# ============================================================

from api.customer_service import (
    CustomerIntelligenceService
)

from api.churn_service import (
    ChurnPredictionService
)

from api.clv_service import (
    CLVPredictionService
)

from api.recommendation_service import (
    RecommendationService
)

from api.explainability_service import (
    ExplainabilityService
)

from api.intelligence_service import (
    UnifiedIntelligenceService
)

from api.schemas import (
    APIInfoResponse,
    CustomerIntelligenceResponse,
    CustomerListResponse,
    CustomerSummaryResponse,
    DatasetStatusResponse,
    HealthResponse,
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    ChurnModelStatusResponse,
    CLVPredictionRequest,
    CLVPredictionResponse,
    CLVModelStatusResponse,
    RecommendationResponse,
    RecommendationModelStatusResponse,
    CustomerExplanationResponse,
    GlobalExplanationResponse,
    ExplainabilityStatusResponse,
    UnifiedCustomerIntelligenceResponse
)


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(

    title=
        "AI-Powered Customer Intelligence API",

    description=
        (
            "REST API for customer churn prediction, "
            "customer value intelligence, recommendations, "
            "customer profiles and SHAP explainability."
        ),

    version=
        "1.5.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "*"
    ],

    allow_credentials=False,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ]
)


# ============================================================
# PATHS
# ============================================================

CUSTOMER_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "dashboard",
    "unified_customer_intelligence.csv"
)

CHURN_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "churn",
    "churn_model.pkl"
)

CLV_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "clv",
    "clv_model.pkl"
)

CLV_VALUE_BANDS_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "clv",
    "clv_value_bands.json"
)

RECOMMENDATION_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "recommendation",
    "recommender.pkl"
)

RECOMMENDATION_BATCH_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "recommendation",
    "predictions",
    "batch_recommendations.csv"
)

CHURN_GLOBAL_SHAP_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "explainability",
    "churn",
    "churn_global_shap_importance.csv"
)

CHURN_LOCAL_SHAP_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "explainability",
    "churn",
    "churn_dashboard_shap_values.csv"
)

CLV_GLOBAL_SHAP_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "explainability",
    "clv",
    "clv_global_shap_importance.csv"
)

CLV_LOCAL_SHAP_PATH = os.path.join(
    PROJECT_ROOT,
    "artifacts",
    "explainability",
    "clv",
    "clv_dashboard_shap_values.csv"
)


# ============================================================
# SERVICES
# ============================================================

customer_service = (
    CustomerIntelligenceService(
        CUSTOMER_DATA_PATH
    )
)

churn_service = (
    ChurnPredictionService(
        model_path=
            CHURN_MODEL_PATH,

        decision_threshold=
            0.50,

        low_risk_threshold=
            0.30,

        high_risk_threshold=
            0.70
    )
)

clv_service = (
    CLVPredictionService(
        model_path=
            CLV_MODEL_PATH,

        value_bands_path=
            CLV_VALUE_BANDS_PATH,

        clip_negative_predictions=
            True
    )
)

recommendation_service = (
    RecommendationService(
        model_path=
            RECOMMENDATION_MODEL_PATH,

        batch_predictions_path=
            RECOMMENDATION_BATCH_PATH
    )
)

explainability_service = (
    ExplainabilityService(
        churn_global_path=
            CHURN_GLOBAL_SHAP_PATH,

        churn_local_path=
            CHURN_LOCAL_SHAP_PATH,

        clv_global_path=
            CLV_GLOBAL_SHAP_PATH,

        clv_local_path=
            CLV_LOCAL_SHAP_PATH
    )
)

intelligence_service = (
    UnifiedIntelligenceService(
        customer_service=
            customer_service,

        recommendation_service=
            recommendation_service,

        explainability_service=
            explainability_service
    )
)


# ============================================================
# ROOT
# ============================================================

@app.get(
    "/",
    response_model=
        APIInfoResponse,
    tags=[
        "General"
    ]
)
def root():

    return {

        "name":
            "AI-Powered Customer Intelligence API",

        "version":
            "1.5.0",

        "description":
            (
                "Integrated API layer for the "
                "AI-Powered Customer Intelligence Platform."
            ),

        "endpoints": [
            "/health",
            "/data/status",
            "/customers",
            "/customers/{customer_id}",
            "/customers/{customer_id}/summary",
            "/models/churn/status",
            "/models/clv/status",
            "/models/recommendation/status",
            "/explainability/status",
            "/predict/churn",
            "/predict/clv",
            "/recommend/{customer_id}",
            "/explain/global/churn",
            "/explain/global/clv",
            "/explain/churn/{customer_id}",
            "/explain/clv/{customer_id}",
            "/intelligence/{customer_id}",
            "/docs"
        ]
    }


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health",
    response_model=
        HealthResponse,
    tags=[
        "General"
    ]
)
def health():

    return {

        "status":
            "healthy",

        "service":
            "customer-intelligence-api",

        "version":
            "1.5.0"
    }


# ============================================================
# DATA STATUS
# ============================================================

@app.get(
    "/data/status",
    response_model=
        DatasetStatusResponse,
    tags=[
        "Data"
    ]
)
def data_status():

    try:

        return (
            customer_service
            .dataset_status()
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(
                error
            )
        )


# ============================================================
# LIST CUSTOMERS
# ============================================================

@app.get(
    "/customers",
    response_model=
        CustomerListResponse,
    tags=[
        "Customers"
    ]
)
def get_customers(

    limit: int = Query(
        default=20,
        ge=1,
        le=100
    ),

    offset: int = Query(
        default=0,
        ge=0
    )
):

    try:

        return (
            customer_service
            .list_customers(
                limit=limit,
                offset=offset
            )
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=503,
            detail=str(
                error
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(
                error
            )
        )


# ============================================================
# CUSTOMER SUMMARY
# ============================================================

@app.get(
    "/customers/{customer_id}/summary",
    response_model=
        CustomerSummaryResponse,
    tags=[
        "Customers"
    ]
)
def get_customer_summary(
    customer_id: str
):

    try:

        result = (
            customer_service
            .get_customer_summary(
                customer_id
            )
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=503,
            detail=str(
                error
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(
                error
            )
        )

    if result is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Customer {customer_id} "
                "was not found."
            )
        )

    return result


# ============================================================
# COMPLETE CUSTOMER
# ============================================================

@app.get(
    "/customers/{customer_id}",
    response_model=
        CustomerIntelligenceResponse,
    tags=[
        "Customers"
    ]
)
def get_customer(
    customer_id: str
):

    try:

        result = (
            customer_service
            .get_customer(
                customer_id
            )
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=503,
            detail=str(
                error
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(
                error
            )
        )

    if result is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Customer {customer_id} "
                "was not found."
            )
        )

    return {

        "customer_id":
            customer_service
            .normalize_customer_id(
                customer_id
            ),

        "intelligence":
            result
    }


# ============================================================
# MODEL STATUS
# ============================================================

@app.get(
    "/models/churn/status",
    response_model=
        ChurnModelStatusResponse,
    tags=[
        "Models"
    ]
)
def churn_model_status():

    return churn_service.model_status()


@app.get(
    "/models/clv/status",
    response_model=
        CLVModelStatusResponse,
    tags=[
        "Models"
    ]
)
def clv_model_status():

    return clv_service.model_status()


@app.get(
    "/models/recommendation/status",
    response_model=
        RecommendationModelStatusResponse,
    tags=[
        "Models"
    ]
)
def recommendation_model_status():

    return (
        recommendation_service
        .model_status()
    )


# ============================================================
# EXPLAINABILITY STATUS
# ============================================================

@app.get(
    "/explainability/status",
    response_model=
        ExplainabilityStatusResponse,
    tags=[
        "Explainability"
    ]
)
def explainability_status():

    return (
        explainability_service
        .status()
    )


# ============================================================
# CHURN PREDICTION
# ============================================================

@app.post(
    "/predict/churn",
    response_model=
        ChurnPredictionResponse,
    tags=[
        "Predictions"
    ]
)
def predict_churn(
    request: ChurnPredictionRequest
):

    try:

        return (
            churn_service
            .predict(
                request.model_dump()
            )
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=503,
            detail=str(
                error
            )
        )

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(
                error
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Churn prediction failed: "
                f"{error}"
            )
        )


# ============================================================
# CLV PREDICTION
# ============================================================

@app.post(
    "/predict/clv",
    response_model=
        CLVPredictionResponse,
    tags=[
        "Predictions"
    ]
)
def predict_clv(
    request: CLVPredictionRequest
):

    try:

        return (
            clv_service
            .predict(
                request.model_dump()
            )
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=503,
            detail=str(
                error
            )
        )

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(
                error
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "CLV prediction failed: "
                f"{error}"
            )
        )


# ============================================================
# RECOMMENDATION
# ============================================================

@app.get(
    "/recommend/{customer_id}",
    response_model=
        RecommendationResponse,
    tags=[
        "Recommendations"
    ]
)
def recommend_customer(

    customer_id: str,

    top_n: int = Query(
        default=5,
        ge=1,
        le=50
    )
):

    try:

        return (
            recommendation_service
            .recommend(
                customer_id=
                    customer_id,

                top_n=
                    top_n
            )
        )

    except LookupError as error:

        raise HTTPException(
            status_code=404,
            detail=str(
                error
            )
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=503,
            detail=str(
                error
            )
        )

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(
                error
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Recommendation generation failed: "
                f"{error}"
            )
        )


# ============================================================
# GLOBAL EXPLANATIONS
# ============================================================

@app.get(
    "/explain/global/churn",
    response_model=
        GlobalExplanationResponse,
    tags=[
        "Explainability"
    ]
)
def explain_global_churn():

    return (
        explainability_service
        .global_churn()
    )


@app.get(
    "/explain/global/clv",
    response_model=
        GlobalExplanationResponse,
    tags=[
        "Explainability"
    ]
)
def explain_global_clv():

    return (
        explainability_service
        .global_clv()
    )


# ============================================================
# CUSTOMER EXPLANATIONS
# ============================================================

@app.get(
    "/explain/churn/{customer_id}",
    response_model=
        CustomerExplanationResponse,
    tags=[
        "Explainability"
    ]
)
def explain_churn_customer(

    customer_id: str,

    top_n: int = Query(
        default=3,
        ge=1,
        le=6
    )
):

    return (
        explainability_service
        .explain_churn_customer(
            customer_id=
                customer_id,

            top_n=
                top_n
        )
    )


@app.get(
    "/explain/clv/{customer_id}",
    response_model=
        CustomerExplanationResponse,
    tags=[
        "Explainability"
    ]
)
def explain_clv_customer(

    customer_id: str,

    top_n: int = Query(
        default=3,
        ge=1,
        le=6
    )
):

    return (
        explainability_service
        .explain_clv_customer(
            customer_id=
                customer_id,

            top_n=
                top_n
        )
    )


# ============================================================
# UNIFIED CUSTOMER INTELLIGENCE
# ============================================================

@app.get(
    "/intelligence/{customer_id}",
    response_model=
        UnifiedCustomerIntelligenceResponse,
    tags=[
        "Intelligence"
    ]
)
def get_unified_intelligence(

    customer_id: str,

    recommendation_top_n: int = Query(
        default=5,
        ge=1,
        le=20
    ),

    explanation_top_n: int = Query(
        default=3,
        ge=1,
        le=6
    )
):

    try:

        result = (
            intelligence_service
            .get_intelligence(
                customer_id=
                    customer_id,

                recommendation_top_n=
                    recommendation_top_n,

                explanation_top_n=
                    explanation_top_n
            )
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=503,
            detail=str(
                error
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to build unified "
                f"customer intelligence: {error}"
            )
        )

    if result is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Customer {customer_id} "
                "was not found."
            )
        )

    return result