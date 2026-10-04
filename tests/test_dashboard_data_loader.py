import pandas as pd

from dashboard.data_loader import (
    filter_dashboard_data,
    calculate_overview_metrics,
    get_unique_values
)


# =============================================================
# TEST DATA
# =============================================================

def create_dashboard_data():

    return pd.DataFrame(
        {
            "Customer ID": [
                101,
                102,
                103,
                104
            ],

            "Customer Segment": [
                "High Value",
                "At Risk",
                "Regular",
                "High Value"
            ],

            "Churn Probability": [
                0.10,
                0.85,
                0.40,
                0.20
            ],

            "Churn Risk": [
                "Low",
                "High",
                "Medium",
                "Low"
            ],

            "Predicted 90-Day Revenue": [
                2000,
                100,
                500,
                3000
            ],

            "CLV Value Band": [
                "High",
                "Low",
                "Medium",
                "High"
            ],

            "Top Recommended Product": [
                "Product 1",
                "Product 2",
                "Product 3",
                "Product 1"
            ]
        }
    )


# =============================================================
# FILTER TEST
# =============================================================

def test_filter_dashboard_data():

    df = (
        create_dashboard_data()
    )

    result = (
        filter_dashboard_data(
            df=
                df,

            segments=[
                "High Value"
            ],

            churn_risks=[
                "Low"
            ],

            clv_bands=[
                "High"
            ]
        )
    )

    assert (
        len(result)
        ==
        2
    )

    assert set(
        result[
            "Customer ID"
        ]
    ) == {
        101,
        104
    }


# =============================================================
# CUSTOMER SEARCH
# =============================================================

def test_customer_search():

    df = (
        create_dashboard_data()
    )

    result = (
        filter_dashboard_data(
            df=
                df,

            customer_search=
                "102"
        )
    )

    assert (
        len(result)
        ==
        1
    )

    assert (
        result[
            "Customer ID"
        ]
        .iloc[0]
        ==
        102
    )


# =============================================================
# KPI TEST
# =============================================================

def test_overview_metrics():

    df = (
        create_dashboard_data()
    )

    metrics = (
        calculate_overview_metrics(
            df
        )
    )

    assert (
        metrics[
            "total_customers"
        ]
        ==
        4
    )

    assert (
        metrics[
            "high_risk_customers"
        ]
        ==
        1
    )

    assert (
        metrics[
            "high_value_customers"
        ]
        ==
        2
    )

    assert (
        metrics[
            "average_predicted_revenue"
        ]
        ==
        1400.0
    )

    assert (
        metrics[
            "total_predicted_revenue"
        ]
        ==
        5600.0
    )


# =============================================================
# UNIQUE FILTER VALUES
# =============================================================

def test_unique_values():

    df = (
        create_dashboard_data()
    )

    values = (
        get_unique_values(
            df,
            "Churn Risk"
        )
    )

    assert values == [
        "High",
        "Low",
        "Medium"
    ]