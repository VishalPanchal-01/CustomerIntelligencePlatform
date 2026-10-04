import pandas as pd

from dashboard.executive_overview import (
    ExecutiveOverview
)


# =============================================================
# TEST DATA
# =============================================================

def create_test_data():

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
                "Low Engagement"
            ],

            "Churn Probability": [
                0.85,
                0.80,
                0.50,
                0.10,
                0.90,
                0.15
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
                100
            ],

            "CLV Value Band": [
                "High",
                "Medium",
                "Medium",
                "High",
                "Low",
                "Low"
            ],

            "Top Recommended Product": [
                "Product A",
                "Product B",
                "Product C",
                "Product D",
                "Product E",
                "Product F"
            ]
        }
    )


# =============================================================
# PRIORITY CLASSIFICATION
# =============================================================

def test_customer_priority():

    overview = (
        ExecutiveOverview()
    )

    result = (
        overview.add_customer_priority(
            create_test_data()
        )
    )

    customer_101 = (
        result[
            result[
                "Customer ID"
            ]
            ==
            101
        ]
        .iloc[0]
    )

    customer_102 = (
        result[
            result[
                "Customer ID"
            ]
            ==
            102
        ]
        .iloc[0]
    )

    customer_104 = (
        result[
            result[
                "Customer ID"
            ]
            ==
            104
        ]
        .iloc[0]
    )

    assert (
        customer_101[
            "Customer Priority"
        ]
        ==
        "Critical"
    )

    assert (
        customer_102[
            "Customer Priority"
        ]
        ==
        "High"
    )

    assert (
        customer_104[
            "Customer Priority"
        ]
        ==
        "Medium"
    )


# =============================================================
# EXECUTIVE METRICS
# =============================================================

def test_executive_metrics():

    overview = (
        ExecutiveOverview()
    )

    metrics = (
        overview.calculate_metrics(
            create_test_data()
        )
    )

    assert (
        metrics[
            "total_customers"
        ]
        ==
        6
    )

    # High-risk customers:
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

    assert (
        metrics[
            "critical_customers"
        ]
        ==
        1
    )

    assert (
        metrics[
            "critical_customer_revenue"
        ]
        ==
        3000.0
    )

    assert (
        metrics[
            "high_priority_customers"
        ]
        ==
        1
    )


# =============================================================
# PRIORITY DISTRIBUTION
# =============================================================

def test_priority_distribution():

    overview = (
        ExecutiveOverview()
    )

    result = (
        overview.priority_distribution(
            create_test_data()
        )
    )

    critical = (
        result[
            result[
                "Customer Priority"
            ]
            ==
            "Critical"
        ][
            "Customers"
        ]
        .iloc[0]
    )

    assert (
        critical
        ==
        1
    )


# =============================================================
# PRIORITY CUSTOMER ORDERING
# =============================================================

def test_priority_customer_ordering():

    overview = (
        ExecutiveOverview()
    )

    result = (
        overview.priority_customers(
            create_test_data(),
            top_n=3
        )
    )

    assert (
        result[
            "Customer ID"
        ]
        .iloc[0]
        ==
        101
    )

    assert (
        result[
            "Customer Priority"
        ]
        .iloc[0]
        ==
        "Critical"
    )


# =============================================================
# BUSINESS ACTION
# =============================================================

def test_business_action_exists():

    overview = (
        ExecutiveOverview()
    )

    result = (
        overview.add_customer_priority(
            create_test_data()
        )
    )

    assert (
        "Recommended Business Action"
        in result.columns
    )

    assert (
        result[
            "Recommended Business Action"
        ]
        .notna()
        .all()
    )


# =============================================================
# SEGMENT PERFORMANCE
# =============================================================

def test_segment_performance():

    overview = (
        ExecutiveOverview()
    )

    result = (
        overview.segment_performance(
            create_test_data()
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
        "High Risk Customers"
        in result.columns
    )

    assert (
        "Critical Customers"
        in result.columns
    )

    assert (
        "Total Predicted Revenue"
        in result.columns
    )