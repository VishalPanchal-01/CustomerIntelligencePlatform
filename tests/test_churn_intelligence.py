import pandas as pd

from dashboard.churn_intelligence import (
    ChurnIntelligence
)


# =============================================================
# TEST DATA
# =============================================================

def create_churn_data():

    return pd.DataFrame(
        {
            "Customer ID": [
                101,
                102,
                103,
                104,
                105,
                106
            ],

            "Customer Segment": [
                "VIP",
                "At Risk",
                "Regular",
                "VIP",
                "Regular",
                "At Risk"
            ],

            "Churn Probability": [
                0.90,
                0.80,
                0.50,
                0.10,
                0.75,
                0.20
            ],

            "Churn Risk": [
                "High",
                "High",
                "Medium",
                "Low",
                "High",
                "Low"
            ],

            "Predicted 90-Day Revenue": [
                3000,
                1000,
                700,
                2500,
                200,
                400
            ],

            "CLV Value Band": [
                "High",
                "Medium",
                "Medium",
                "High",
                "Low",
                "Low"
            ],

            "Recency": [
                100,
                120,
                60,
                10,
                150,
                20
            ],

            "Frequency": [
                2,
                3,
                5,
                15,
                1,
                10
            ],

            "Monetary": [
                3000,
                1000,
                700,
                5000,
                200,
                1000
            ],

            "TotalItems": [
                100,
                40,
                30,
                200,
                5,
                60
            ],

            "AverageOrderValue": [
                1500,
                333.33,
                140,
                333.33,
                200,
                100
            ],

            "Tenure": [
                500,
                400,
                300,
                600,
                100,
                350
            ],

            "Top Recommended Product": [
                "A",
                "B",
                "C",
                "D",
                "E",
                "F"
            ]
        }
    )


# =============================================================
# METRICS
# =============================================================

def test_churn_metrics():

    dashboard = (
        ChurnIntelligence()
    )

    metrics = (
        dashboard.calculate_metrics(
            create_churn_data()
        )
    )

    assert (
        metrics[
            "total_customers"
        ]
        ==
        6
    )

    assert (
        metrics[
            "high_risk_customers"
        ]
        ==
        3
    )

    assert (
        metrics[
            "medium_risk_customers"
        ]
        ==
        1
    )

    assert (
        metrics[
            "low_risk_customers"
        ]
        ==
        2
    )

    # High-risk revenue:
    # 101 = 3000
    # 102 = 1000
    # 105 = 200

    assert (
        metrics[
            "revenue_at_risk"
        ]
        ==
        4200.0
    )

    # High-value + high-risk:
    # customer 101 only

    assert (
        metrics[
            "high_value_at_risk"
        ]
        ==
        1
    )


# =============================================================
# RISK DISTRIBUTION
# =============================================================

def test_risk_distribution():

    dashboard = (
        ChurnIntelligence()
    )

    result = (
        dashboard.risk_distribution(
            create_churn_data()
        )
    )

    high = (
        result[
            result[
                "Churn Risk"
            ]
            ==
            "High"
        ][
            "Customers"
        ]
        .iloc[0]
    )

    assert (
        high
        ==
        3
    )


# =============================================================
# CHURN BY SEGMENT
# =============================================================

def test_churn_by_segment():

    dashboard = (
        ChurnIntelligence()
    )

    result = (
        dashboard.churn_by_segment(
            create_churn_data()
        )
    )

    assert (
        "Customer Segment"
        in result.columns
    )

    assert (
        "Customers"
        in result.columns
    )

    assert (
        "Average Churn Probability"
        in result.columns
    )

    assert (
        "High Risk Customers"
        in result.columns
    )

    assert (
        "High Risk %"
        in result.columns
    )


# =============================================================
# REVENUE AT RISK
# =============================================================

def test_revenue_at_risk_by_segment():

    dashboard = (
        ChurnIntelligence()
    )

    result = (
        dashboard.revenue_at_risk_by_segment(
            create_churn_data()
        )
    )

    assert (
        "Customer Segment"
        in result.columns
    )

    assert (
        "High Risk Customers"
        in result.columns
    )

    assert (
        "Revenue at Risk"
        in result.columns
    )

    assert (
        result[
            "Revenue at Risk"
        ]
        .sum()
        ==
        4200.0
    )


# =============================================================
# HIGH VALUE AT RISK
# =============================================================

def test_high_value_customers_at_risk():

    dashboard = (
        ChurnIntelligence()
    )

    result = (
        dashboard.high_value_customers_at_risk(
            create_churn_data()
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
        101
    )


# =============================================================
# RETENTION PRIORITY
# =============================================================

def test_retention_priority():

    dashboard = (
        ChurnIntelligence()
    )

    result = (
        dashboard.retention_priority_customers(
            create_churn_data(),
            top_n=3
        )
    )

    assert (
        len(result)
        ==
        3
    )

    assert (
        "Retention Priority Score"
        in result.columns
    )

    assert (
        result[
            "Retention Priority Score"
        ]
        .is_monotonic_decreasing
    )


# =============================================================
# DATA PREPARATION
# =============================================================

def test_prepare_data():

    dashboard = (
        ChurnIntelligence()
    )

    df = (
        create_churn_data()
    )

    df[
        "Recency"
    ] = (
        df[
            "Recency"
        ]
        .astype(str)
    )

    result = (
        dashboard.prepare_data(
            df
        )
    )

    assert pd.api.types.is_numeric_dtype(
        result[
            "Recency"
        ]
    )

    assert pd.api.types.is_numeric_dtype(
        result[
            "Churn Probability"
        ]
    )