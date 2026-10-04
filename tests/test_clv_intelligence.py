import pandas as pd
import pytest

from dashboard.clv_intelligence import (
    CLVIntelligence
)


# =============================================================
# TEST DATA
# =============================================================

def create_clv_data():

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

            "Predicted 90-Day Revenue": [
                3000,
                200,
                900,
                4000,
                700,
                300
            ],

            "CLV Value Band": [
                "High",
                "Low",
                "Medium",
                "High",
                "Medium",
                "Low"
            ],

            "Churn Probability": [
                0.85,
                0.80,
                0.40,
                0.10,
                0.60,
                0.75
            ],

            "Churn Risk": [
                "High",
                "High",
                "Medium",
                "Low",
                "Medium",
                "High"
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
# PREPARE DATA
# =============================================================

def test_prepare_data():

    dashboard = (
        CLVIntelligence()
    )

    df = (
        create_clv_data()
    )

    df[
        "Predicted 90-Day Revenue"
    ] = (
        df[
            "Predicted 90-Day Revenue"
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
            "Predicted 90-Day Revenue"
        ]
    )


# =============================================================
# METRICS
# =============================================================

def test_clv_metrics():

    dashboard = (
        CLVIntelligence()
    )

    metrics = (
        dashboard.calculate_metrics(
            create_clv_data()
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
            "total_predicted_revenue"
        ]
        ==
        9100.0
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
            "high_value_revenue"
        ]
        ==
        7000.0
    )

    # Customer 101:
    # High CLV + High Churn Risk
    assert (
        metrics[
            "high_value_at_risk"
        ]
        ==
        1
    )


# =============================================================
# VALUE BAND DISTRIBUTION
# =============================================================

def test_value_band_distribution():

    dashboard = (
        CLVIntelligence()
    )

    result = (
        dashboard.value_band_distribution(
            create_clv_data()
        )
    )

    assert (
        result[
            "Customers"
        ]
        .sum()
        ==
        6
    )

    high = (
        result[
            result[
                "CLV Value Band"
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
        2
    )

    # Individual percentages are rounded,
    # therefore allow a very small tolerance.

    assert (
        result[
            "Customer Percentage"
        ]
        .sum()
        ==
        pytest.approx(
            100.0,
            abs=0.02
        )
    )


# =============================================================
# REVENUE BY VALUE BAND
# =============================================================

def test_revenue_by_value_band():

    dashboard = (
        CLVIntelligence()
    )

    result = (
        dashboard.revenue_by_value_band(
            create_clv_data()
        )
    )

    assert (
        "Revenue Contribution %"
        in result.columns
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # Individual category percentages are rounded to
    # two decimal places.
    #
    # Example:
    #
    # High    = 76.92
    # Medium  = 17.58
    # Low     = 5.49
    #
    # Rounded total = 99.99
    #
    # This is normal floating-point / percentage rounding.
    # ---------------------------------------------------------

    contribution_total = (
        result[
            "Revenue Contribution %"
        ]
        .sum()
    )

    assert (
        contribution_total
        ==
        pytest.approx(
            100.0,
            abs=0.02
        )
    )

    high = (
        result[
            result[
                "CLV Value Band"
            ]
            ==
            "High"
        ]
        .iloc[0]
    )

    assert (
        high[
            "Total Predicted Revenue"
        ]
        ==
        7000.0
    )


# =============================================================
# CLV BY SEGMENT
# =============================================================

def test_clv_by_segment():

    dashboard = (
        CLVIntelligence()
    )

    result = (
        dashboard.clv_by_segment(
            create_clv_data()
        )
    )

    assert (
        "Customer Segment"
        in result.columns
    )

    assert (
        "Average Predicted Revenue"
        in result.columns
    )

    assert (
        "Total Predicted Revenue"
        in result.columns
    )

    vip = (
        result[
            result[
                "Customer Segment"
            ]
            ==
            "VIP"
        ]
        .iloc[0]
    )

    assert (
        vip[
            "Total Predicted Revenue"
        ]
        ==
        7000.0
    )


# =============================================================
# HIGH VALUE AT RISK
# =============================================================

def test_high_value_at_risk():

    dashboard = (
        CLVIntelligence()
    )

    result = (
        dashboard.high_value_at_risk(
            create_clv_data()
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
# CUSTOMER VALUE RANKING
# =============================================================

def test_customer_value_ranking():

    dashboard = (
        CLVIntelligence()
    )

    result = (
        dashboard.customer_value_ranking(
            create_clv_data(),
            top_n=3
        )
    )

    assert (
        len(result)
        ==
        3
    )

    # Customer 104 has highest
    # predicted 90-day revenue = 4000.

    assert (
        result[
            "Customer ID"
        ]
        .iloc[0]
        ==
        104
    )

    assert (
        result[
            "Customer Value Rank"
        ]
        .tolist()
        ==
        [
            1,
            2,
            3
        ]
    )

    assert (
        result[
            "Predicted 90-Day Revenue"
        ]
        .is_monotonic_decreasing
    )


# =============================================================
# REVENUE CONCENTRATION
# =============================================================

def test_revenue_concentration():

    dashboard = (
        CLVIntelligence()
    )

    result = (
        dashboard.revenue_concentration(
            create_clv_data()
        )
    )

    assert (
        "Customer Percentile"
        in result.columns
    )

    assert (
        "Cumulative Revenue %"
        in result.columns
    )

    # Final cumulative revenue should
    # reach approximately 100%.

    assert (
        result[
            "Cumulative Revenue %"
        ]
        .iloc[-1]
        ==
        pytest.approx(
            100.0,
            abs=0.01
        )
    )

    # Cumulative percentage should never decrease.

    assert (
        result[
            "Cumulative Revenue %"
        ]
        .is_monotonic_increasing
    )


# =============================================================
# TOP CUSTOMER REVENUE SHARE
# =============================================================

def test_top_customer_revenue_share():

    dashboard = (
        CLVIntelligence()
    )

    share = (
        dashboard.top_customer_revenue_share(
            create_clv_data(),
            top_percentage=20
        )
    )

    assert (
        share
        >
        0
    )

    assert (
        share
        <=
        100
    )


# =============================================================
# REVENUE TOTAL CONSISTENCY
# =============================================================

def test_revenue_total_consistency():

    dashboard = (
        CLVIntelligence()
    )

    df = (
        create_clv_data()
    )

    band_result = (
        dashboard.revenue_by_value_band(
            df
        )
    )

    expected_total = (
        df[
            "Predicted 90-Day Revenue"
        ]
        .sum()
    )

    calculated_total = (
        band_result[
            "Total Predicted Revenue"
        ]
        .sum()
    )

    assert (
        calculated_total
        ==
        expected_total
    )