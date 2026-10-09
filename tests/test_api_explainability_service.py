import os

import pandas as pd

from api.explainability_service import (
    ExplainabilityService
)


# ============================================================
# TEST FILES
# ============================================================

def create_files(
    tmp_path
):

    churn_global_path = os.path.join(
        tmp_path,
        "churn_global.csv"
    )

    clv_global_path = os.path.join(
        tmp_path,
        "clv_global.csv"
    )

    churn_local_path = os.path.join(
        tmp_path,
        "churn_local.csv"
    )

    clv_local_path = os.path.join(
        tmp_path,
        "clv_local.csv"
    )

    global_data = pd.DataFrame(
        {
            "Feature": [
                "Recency",
                "Frequency",
                "Monetary"
            ],

            "Mean Absolute SHAP": [
                0.5,
                0.3,
                0.2
            ]
        }
    )

    global_data.to_csv(
        churn_global_path,
        index=False
    )

    global_data.to_csv(
        clv_global_path,
        index=False
    )

    local_data = pd.DataFrame(
        {
            "Customer ID": [
                101,
                101,
                101,
                102
            ],

            "Feature": [
                "Recency",
                "Frequency",
                "Monetary",
                "Recency"
            ],

            "Feature Value": [
                100,
                2,
                500,
                50
            ],

            "SHAP Value": [
                0.5,
                0.3,
                -0.4,
                0.1
            ],

            "Absolute SHAP": [
                0.5,
                0.3,
                0.4,
                0.1
            ],

            "Impact Rank": [
                1,
                3,
                2,
                1
            ]
        }
    )

    local_data.to_csv(
        churn_local_path,
        index=False
    )

    local_data.to_csv(
        clv_local_path,
        index=False
    )

    return {
        "churn_global":
            churn_global_path,

        "churn_local":
            churn_local_path,

        "clv_global":
            clv_global_path,

        "clv_local":
            clv_local_path
    }


# ============================================================
# SERVICE
# ============================================================

def create_service(
    tmp_path
):

    paths = create_files(
        tmp_path
    )

    return ExplainabilityService(
        churn_global_path=
            paths[
                "churn_global"
            ],

        churn_local_path=
            paths[
                "churn_local"
            ],

        clv_global_path=
            paths[
                "clv_global"
            ],

        clv_local_path=
            paths[
                "clv_local"
            ]
    )


# ============================================================
# GLOBAL CHURN
# ============================================================

def test_global_churn(
    tmp_path
):

    service = create_service(
        tmp_path
    )

    result = (
        service.global_churn()
    )

    assert (
        result[
            "available"
        ]
        is True
    )

    assert (
        result[
            "feature_importance"
        ][0][
            "feature"
        ]
        ==
        "Recency"
    )


# ============================================================
# CUSTOMER CHURN EXPLANATION
# ============================================================

def test_customer_churn_explanation(
    tmp_path
):

    service = create_service(
        tmp_path
    )

    result = (
        service
        .explain_churn_customer(
            "101",
            top_n=2
        )
    )

    assert (
        result[
            "explanation_available"
        ]
        is True
    )

    assert (
        len(
            result[
                "top_positive_drivers"
            ]
        )
        ==
        2
    )

    assert (
        result[
            "top_negative_drivers"
        ][0][
            "feature"
        ]
        ==
        "Monetary"
    )


# ============================================================
# UNKNOWN CUSTOMER
# ============================================================

def test_unknown_customer(
    tmp_path
):

    service = create_service(
        tmp_path
    )

    result = (
        service
        .explain_churn_customer(
            "999"
        )
    )

    assert (
        result[
            "explanation_available"
        ]
        is False
    )

    assert (
        result[
            "all_contributions"
        ]
        ==
        []
    )


# ============================================================
# CLV CUSTOMER
# ============================================================

def test_customer_clv_explanation(
    tmp_path
):

    service = create_service(
        tmp_path
    )

    result = (
        service
        .explain_clv_customer(
            "101"
        )
    )

    assert (
        result[
            "model"
        ]
        ==
        "predicted_90_day_revenue"
    )

    assert (
        result[
            "explanation_available"
        ]
        is True
    )


# ============================================================
# STATUS
# ============================================================

def test_status(
    tmp_path
):

    service = create_service(
        tmp_path
    )

    result = (
        service.status()
    )

    assert (
        result[
            "churn_global_available"
        ]
        is True
    )

    assert (
        result[
            "clv_global_available"
        ]
        is True
    )

    assert (
        result[
            "churn_explained_customers"
        ]
        ==
        2
    )

    assert (
        result[
            "clv_explained_customers"
        ]
        ==
        2
    )