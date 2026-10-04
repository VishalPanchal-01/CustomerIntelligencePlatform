import pandas as pd

from dashboard.customer_explorer import (
    CustomerExplorer
)


# ============================================================
# INSTANCE
# ============================================================

def create_explorer():

    return CustomerExplorer()


# ============================================================
# CUSTOMER ID NORMALIZATION
# ============================================================

def test_customer_id_normalization():

    explorer = create_explorer()

    assert (
        explorer.normalize_customer_id(
            17850
        )
        ==
        "17850"
    )

    assert (
        explorer.normalize_customer_id(
            17850.0
        )
        ==
        "17850"
    )

    assert (
        explorer.normalize_customer_id(
            "17850.0"
        )
        ==
        "17850"
    )

    assert (
        explorer.normalize_customer_id(
            "ABC123"
        )
        ==
        "ABC123"
    )


# ============================================================
# GET CUSTOMER
# ============================================================

def test_get_customer_with_numeric_string_id():

    explorer = create_explorer()

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

    customer = (
        explorer
        .get_customer(
            data,
            "101"
        )
    )

    assert customer is not None

    assert (
        customer[
            "Churn Probability"
        ]
        ==
        0.8
    )


# ============================================================
# SHAP SUMMARY
# ============================================================

def test_shap_driver_summary():

    explorer = create_explorer()

    data = pd.DataFrame(
        {
            "Feature": [
                "Recency",
                "Frequency",
                "Monetary",
                "Tenure"
            ],

            "Feature Value": [
                100,
                2,
                5000,
                300
            ],

            "SHAP Value": [
                0.50,
                0.30,
                -0.40,
                -0.10
            ],

            "Absolute SHAP": [
                0.50,
                0.30,
                0.40,
                0.10
            ]
        }
    )

    result = (
        explorer
        .shap_driver_summary(
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
# EMPTY SHAP SUMMARY
# ============================================================

def test_empty_shap_driver_summary():

    explorer = create_explorer()

    result = (
        explorer
        .shap_driver_summary(
            pd.DataFrame()
        )
    )

    assert (
        result[
            "positive"
        ]
        .empty
    )

    assert (
        result[
            "negative"
        ]
        .empty
    )


# ============================================================
# ABSOLUTE SHAP AUTO CREATION
# ============================================================

def test_shap_summary_creates_absolute_values():

    explorer = create_explorer()

    data = pd.DataFrame(
        {
            "Feature": [
                "Recency",
                "Frequency"
            ],

            "SHAP Value": [
                0.6,
                -0.4
            ]
        }
    )

    result = (
        explorer
        .shap_driver_summary(
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
        "Frequency"
    )