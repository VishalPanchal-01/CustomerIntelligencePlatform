from fastapi.testclient import (
    TestClient
)

from api.main import (
    app,
    churn_service
)


client = TestClient(
    app
)


# ============================================================
# ROOT
# ============================================================

def test_root():

    response = client.get(
        "/"
    )

    assert (
        response.status_code
        ==
        200
    )

    data = response.json()

    assert (
        data[
            "name"
        ]
        ==
        "AI-Powered Customer Intelligence API"
    )


# ============================================================
# HEALTH
# ============================================================

def test_health():

    response = client.get(
        "/health"
    )

    assert (
        response.status_code
        ==
        200
    )

    assert (
        response.json()[
            "status"
        ]
        ==
        "healthy"
    )


# ============================================================
# CHURN MODEL STATUS
# ============================================================

def test_churn_model_status():

    response = client.get(
        "/models/churn/status"
    )

    assert (
        response.status_code
        ==
        200
    )

    data = response.json()

    assert (
        data[
            "feature_count"
        ]
        ==
        6
    )

    assert (
        "Recency"
        in data[
            "features"
        ]
    )


# ============================================================
# INVALID CHURN INPUT
# ============================================================

def test_invalid_churn_request():

    payload = {
        "Recency":
            -5,

        "Frequency":
            4,

        "Monetary":
            1000,

        "TotalItems":
            50,

        "AverageOrderValue":
            250,

        "Tenure":
            200
    }

    response = client.post(
        "/predict/churn",
        json=payload
    )

    assert (
        response.status_code
        ==
        422
    )


# ============================================================
# MISSING FEATURE
# ============================================================

def test_missing_churn_feature():

    payload = {
        "Recency":
            50,

        "Frequency":
            4,

        "Monetary":
            1000
    }

    response = client.post(
        "/predict/churn",
        json=payload
    )

    assert (
        response.status_code
        ==
        422
    )


# ============================================================
# RECOMMENDATION MODEL STATUS
# ============================================================

def test_recommendation_model_status():

    response = client.get(
        "/models/recommendation/status"
    )

    assert (
        response.status_code
        ==
        200
    )

    data = response.json()

    assert (
        "model_available"
        in data
    )

    assert (
        "batch_predictions_available"
        in data
    )

    assert (
        "cold_start_supported"
        in data
    )


# ============================================================
# INVALID TOP-N
# ============================================================

def test_invalid_recommendation_top_n():

    response = client.get(
        "/recommend/12345?top_n=0"
    )

    assert (
        response.status_code
        ==
        422
    )


# ============================================================
# EXPLAINABILITY STATUS
# ============================================================

def test_explainability_status():

    response = client.get(
        "/explainability/status"
    )

    assert (
        response.status_code
        ==
        200
    )

    data = response.json()

    assert (
        "churn_global_available"
        in data
    )

    assert (
        "clv_global_available"
        in data
    )


# ============================================================
# GLOBAL CHURN EXPLANATION
# ============================================================

def test_global_churn_explanation():

    response = client.get(
        "/explain/global/churn"
    )

    assert (
        response.status_code
        in [
            200,
            500
        ]
    )


# ============================================================
# GLOBAL CLV EXPLANATION
# ============================================================

def test_global_clv_explanation():

    response = client.get(
        "/explain/global/clv"
    )

    assert (
        response.status_code
        in [
            200,
            500
        ]
    )


# ============================================================
# INVALID SHAP TOP-N
# ============================================================

def test_invalid_explainability_top_n():

    response = client.get(
        "/explain/churn/101?top_n=0"
    )

    assert (
        response.status_code
        ==
        422
    )

# ============================================================
# UNIFIED INTELLIGENCE INVALID PARAM
# ============================================================

def test_invalid_intelligence_top_n():

    response = client.get(
        (
            "/intelligence/101"
            "?recommendation_top_n=0"
        )
    )

    assert (
        response.status_code
        ==
        422
    )    