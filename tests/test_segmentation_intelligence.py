import pandas as pd

from dashboard.segmentation_intelligence import (
    SegmentationIntelligence
)


# =============================================================
# TEST DATA
# =============================================================

def create_segmentation_data():

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

            "Recency": [
                10,
                120,
                50,
                5,
                70,
                100
            ],

            "Frequency": [
                15,
                2,
                6,
                20,
                5,
                3
            ],

            "Monetary": [
                5000,
                500,
                1500,
                7000,
                1200,
                700
            ],

            "TotalItems": [
                300,
                20,
                90,
                500,
                70,
                40
            ],

            "AverageOrderValue": [
                333.33,
                250.00,
                250.00,
                350.00,
                240.00,
                233.33
            ],

            "Tenure": [
                500,
                200,
                300,
                600,
                280,
                250
            ],

            "Churn Probability": [
                0.10,
                0.85,
                0.40,
                0.15,
                0.60,
                0.80
            ],

            "Churn Risk": [
                "Low",
                "High",
                "Medium",
                "Low",
                "Medium",
                "High"
            ],

            "Predicted 90-Day Revenue": [
                3000,
                300,
                800,
                4000,
                700,
                400
            ],

            "CLV Value Band": [
                "High",
                "Low",
                "Medium",
                "High",
                "Medium",
                "Low"
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
        SegmentationIntelligence()
    )

    df = (
        create_segmentation_data()
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


# =============================================================
# METRICS
# =============================================================

def test_segmentation_metrics():

    dashboard = (
        SegmentationIntelligence()
    )

    metrics = (
        dashboard.calculate_metrics(
            create_segmentation_data()
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
            "segment_count"
        ]
        ==
        3
    )

    assert (
        metrics[
            "highest_revenue_segment"
        ]
        ==
        "VIP"
    )


# =============================================================
# DISTRIBUTION
# =============================================================

def test_segment_distribution():

    dashboard = (
        SegmentationIntelligence()
    )

    result = (
        dashboard.segment_distribution(
            create_segmentation_data()
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

    assert abs(
        result[
            "Customer Percentage"
        ]
        .sum()
        -
        100.0
    ) < 0.05


# =============================================================
# SEGMENT PROFILE
# =============================================================

def test_segment_profile():

    dashboard = (
        SegmentationIntelligence()
    )

    result = (
        dashboard.segment_profile(
            create_segmentation_data()
        )
    )

    assert (
        "Customer Segment"
        in result.columns
    )

    assert (
        "Recency"
        in result.columns
    )

    assert (
        "Frequency"
        in result.columns
    )

    assert (
        "Monetary"
        in result.columns
    )


# =============================================================
# BUSINESS PERFORMANCE
# =============================================================

def test_segment_business_performance():

    dashboard = (
        SegmentationIntelligence()
    )

    result = (
        dashboard.segment_business_performance(
            create_segmentation_data()
        )
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
        "High Value Customers"
        in result.columns
    )

    assert (
        "Average Churn Probability"
        in result.columns
    )

    assert (
        "Total Predicted Revenue"
        in result.columns
    )


# =============================================================
# REVENUE CONTRIBUTION
# =============================================================

def test_revenue_contribution():

    dashboard = (
        SegmentationIntelligence()
    )

    result = (
        dashboard.revenue_contribution(
            create_segmentation_data()
        )
    )

    assert (
        "Revenue Contribution %"
        in result.columns
    )

    assert round(
        result[
            "Revenue Contribution %"
        ]
        .sum(),
        2
    ) == 100.0

    # VIP:
    # 3000 + 4000 = 7000

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
# CHURN EXPOSURE
# =============================================================

def test_churn_exposure():

    dashboard = (
        SegmentationIntelligence()
    )

    result = (
        dashboard.churn_exposure(
            create_segmentation_data()
        )
    )

    assert (
        "Customer Segment"
        in result.columns
    )

    assert (
        "Churn Risk"
        in result.columns
    )

    assert (
        "Customers"
        in result.columns
    )


# =============================================================
# CLV COMPOSITION
# =============================================================

def test_clv_composition():

    dashboard = (
        SegmentationIntelligence()
    )

    result = (
        dashboard.clv_composition(
            create_segmentation_data()
        )
    )

    assert (
        "CLV Value Band"
        in result.columns
    )

    assert (
        result[
            "Customers"
        ]
        .sum()
        ==
        6
    )


# =============================================================
# SEGMENT ACTIONS
# =============================================================

def test_segment_actions():

    dashboard = (
        SegmentationIntelligence()
    )

    result = (
        dashboard.segment_actions(
            create_segmentation_data()
        )
    )

    assert (
        "Customer Segment"
        in result.columns
    )

    assert (
        "Recommended Segment Strategy"
        in result.columns
    )

    assert (
        result[
            "Recommended Segment Strategy"
        ]
        .notna()
        .all()
    )


# =============================================================
# SEGMENT CUSTOMER EXPLORER
# =============================================================

def test_segment_customers():

    dashboard = (
        SegmentationIntelligence()
    )

    result = (
        dashboard.segment_customers(
            create_segmentation_data(),
            segment="VIP",
            top_n=10
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