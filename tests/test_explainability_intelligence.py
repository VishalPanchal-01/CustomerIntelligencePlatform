import pandas as pd

from dashboard.explainability_intelligence import (
    ExplainabilityIntelligence
)


# ============================================================
# INSTANCE
# ============================================================

def create_dashboard():

    return ExplainabilityIntelligence(
        project_root="."
    )


# ============================================================
# CUSTOMER ID NORMALIZATION
# ============================================================

def test_customer_id_normalization():

    dashboard = create_dashboard()

    assert (
        dashboard.normalize_customer_id(
            17850
        )
        ==
        "17850"
    )

    assert (
        dashboard.normalize_customer_id(
            17850.0
        )
        ==
        "17850"
    )

    assert (
        dashboard.normalize_customer_id(
            "17850.0"
        )
        ==
        "17850"
    )

    assert (
        dashboard.normalize_customer_id(
            "A101"
        )
        ==
        "A101"
    )


# ============================================================
# GLOBAL IMPORTANCE
# ============================================================

def test_prepare_global_importance():

    dashboard = create_dashboard()

    data = pd.DataFrame(
        {
            "Feature": [
                "Recency",
                "Frequency",
                "Monetary"
            ],

            "Mean Absolute SHAP": [
                0.3,
                0.5,
                0.1
            ]
        }
    )

    result = (
        dashboard
        .prepare_global_importance(
            data,
            "Churn"
        )
    )

    assert (
        result.iloc[0][
            "Feature"
        ]
        ==
        "Frequency"
    )

    assert (
        result.iloc[0][
            "Model"
        ]
        ==
        "Churn"
    )


# ============================================================
# NORMALIZED IMPORTANCE
# ============================================================

def test_normalized_importance_comparison():

    dashboard = create_dashboard()

    churn = pd.DataFrame(
        {
            "Feature": [
                "Recency",
                "Frequency"
            ],

            "Mean Absolute SHAP": [
                0.5,
                0.25
            ]
        }
    )

    clv = pd.DataFrame(
        {
            "Feature": [
                "Monetary",
                "Frequency"
            ],

            "Mean Absolute SHAP": [
                1000,
                500
            ]
        }
    )

    result = (
        dashboard
        .normalized_importance_comparison(
            churn,
            clv
        )
    )

    churn_result = (
        result[
            result[
                "Model"
            ]
            ==
            "Churn"
        ]
    )

    clv_result = (
        result[
            result[
                "Model"
            ]
            ==
            "Predicted 90-Day Revenue"
        ]
    )

    assert (
        churn_result[
            "Normalized Importance"
        ]
        .max()
        ==
        100
    )

    assert (
        clv_result[
            "Normalized Importance"
        ]
        .max()
        ==
        100
    )


# ============================================================
# AVAILABLE CUSTOMERS
# ============================================================

def test_available_customers_prefers_common():

    dashboard = create_dashboard()

    churn = pd.DataFrame(
        {
            "Customer ID": [
                "101",
                "102",
                "103"
            ]
        }
    )

    clv = pd.DataFrame(
        {
            "Customer ID": [
                "102",
                "103",
                "104"
            ]
        }
    )

    result = (
        dashboard
        .available_customers(
            churn,
            clv
        )
    )

    assert result == [
        "102",
        "103"
    ]


# ============================================================
# CUSTOMER SHAP DATA
# ============================================================

def test_customer_shap_data():

    dashboard = create_dashboard()

    data = pd.DataFrame(
        {
            "Customer ID": [
                "101",
                "101",
                "102"
            ],

            "Feature": [
                "Recency",
                "Frequency",
                "Recency"
            ],

            "Feature Value": [
                100,
                2,
                50
            ],

            "SHAP Value": [
                0.4,
                -0.2,
                0.1
            ],

            "Absolute SHAP": [
                0.4,
                0.2,
                0.1
            ]
        }
    )

    result = (
        dashboard
        .customer_shap_data(
            data,
            101
        )
    )

    assert (
        len(result)
        ==
        2
    )

    assert (
        result.iloc[0][
            "Feature"
        ]
        ==
        "Recency"
    )


# ============================================================
# DRIVER SUMMARY
# ============================================================

def test_driver_summary():

    dashboard = create_dashboard()

    data = pd.DataFrame(
        {
            "Feature": [
                "Recency",
                "Frequency",
                "Monetary",
                "Tenure"
            ],

            "SHAP Value": [
                0.5,
                0.3,
                -0.4,
                -0.2
            ],

            "Absolute SHAP": [
                0.5,
                0.3,
                0.4,
                0.2
            ]
        }
    )

    result = (
        dashboard
        .driver_summary(
            data,
            top_n=1
        )
    )

    assert (
        result[
            "positive"
        ]
        .iloc[0][
            "Feature"
        ]
        ==
        "Recency"
    )

    assert (
        result[
            "negative"
        ]
        .iloc[0][
            "Feature"
        ]
        ==
        "Monetary"
    )


# ============================================================
# CUSTOMER RECORD
# ============================================================

def test_find_customer_record():

    dashboard = create_dashboard()

    data = pd.DataFrame(
        {
            "Customer ID": [
                101.0,
                102.0
            ],

            "Churn Probability": [
                0.8,
                0.2
            ]
        }
    )

    result = (
        dashboard
        .find_customer_record(
            data,
            "101"
        )
    )

    assert result is not None

    assert (
        result[
            "Churn Probability"
        ]
        ==
        0.8
    )