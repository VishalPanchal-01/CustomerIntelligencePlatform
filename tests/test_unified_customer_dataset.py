import pandas as pd

from src.dashboard.unified_customer_dataset import (
    UnifiedCustomerDatasetBuilder
)


# =============================================================
# TEST DATA
# =============================================================

def create_segmentation_data():

    return pd.DataFrame(
        {
            # Old name intentionally used
            # to verify backward compatibility.
            "CustomerID": [
                101,
                102,
                103
            ],

            "SegmentName": [
                "High Value Loyal",
                "At Risk",
                "Regular Customer"
            ],

            "Recency": [
                10,
                100,
                35
            ],

            "Frequency": [
                12,
                2,
                6
            ],

            "Monetary": [
                5000,
                500,
                2000
            ],

            "TotalItems": [
                200,
                20,
                90
            ],

            "AverageOrderValue": [
                416.67,
                250.00,
                333.33
            ],

            "Tenure": [
                500,
                150,
                300
            ]
        }
    )


def create_churn_data():

    return pd.DataFrame(
        {
            "CustomerID": [
                101,
                102,
                103
            ],

            "ChurnProbability": [
                0.10,
                0.85,
                0.30
            ],

            "ChurnPrediction": [
                0,
                1,
                0
            ],

            "ChurnRisk": [
                "Low",
                "High",
                "Medium"
            ]
        }
    )


def create_clv_data():

    return pd.DataFrame(
        {
            "CustomerID": [
                101,
                102,
                103
            ],

            "PredictedFutureRevenue": [
                2500,
                150,
                900
            ],

            "CLVValueBand": [
                "High",
                "Low",
                "Medium"
            ],

            "PredictionHorizonDays": [
                90,
                90,
                90
            ]
        }
    )


def create_recommendation_data():

    return pd.DataFrame(
        {
            "CustomerID": [
                101,
                101,
                102,
                102,
                103
            ],

            "StockCode": [
                "P1",
                "P2",
                "P3",
                "P4",
                "P5"
            ],

            "Description": [
                "Product 1",
                "Product 2",
                "Product 3",
                "Product 4",
                "Product 5"
            ],

            "Score": [
                0.95,
                0.85,
                0.91,
                0.75,
                0.88
            ],

            "Rank": [
                1,
                2,
                1,
                2,
                1
            ],

            "RecommendationSource": [
                "Hybrid",
                "Hybrid",
                "Hybrid",
                "Hybrid",
                "Hybrid"
            ]
        }
    )


# =============================================================
# CUSTOMER ID NORMALIZATION
# =============================================================

def test_customer_id_normalization():

    builder = (
        UnifiedCustomerDatasetBuilder()
    )

    df = pd.DataFrame(
        {
            "CustomerID": [
                101,
                102
            ]
        }
    )

    result = (
        builder.normalize_customer_id(
            df
        )
    )

    assert (
        "Customer ID"
        in result.columns
    )

    assert (
        "CustomerID"
        not in result.columns
    )


# =============================================================
# UNIFIED DATASET
# =============================================================

def test_unified_customer_dataset():

    builder = (
        UnifiedCustomerDatasetBuilder()
    )

    result = (
        builder.build(
            segmentation_df=
                create_segmentation_data(),

            churn_df=
                create_churn_data(),

            clv_df=
                create_clv_data(),

            recommendation_df=
                create_recommendation_data()
        )
    )

    # ---------------------------------------------
    # Customer ID standard
    # ---------------------------------------------

    assert (
        "Customer ID"
        in result.columns
    )

    assert (
        "CustomerID"
        not in result.columns
    )

    # ---------------------------------------------
    # One customer = one row
    # ---------------------------------------------

    assert (
        len(result)
        ==
        3
    )

    assert (
        result[
            "Customer ID"
        ]
        .nunique()
        ==
        3
    )

    # ---------------------------------------------
    # Segmentation
    # ---------------------------------------------

    assert (
        "Customer Segment"
        in result.columns
    )

    # ---------------------------------------------
    # Churn
    # ---------------------------------------------

    assert (
        "Churn Probability"
        in result.columns
    )

    assert (
        "Churn Prediction"
        in result.columns
    )

    assert (
        "Churn Risk"
        in result.columns
    )

    # ---------------------------------------------
    # CLV
    # ---------------------------------------------

    assert (
        "Predicted 90-Day Revenue"
        in result.columns
    )

    assert (
        "CLV Value Band"
        in result.columns
    )

    # ---------------------------------------------
    # Recommendations
    # ---------------------------------------------

    assert (
        "Top Recommended Product"
        in result.columns
    )

    assert (
        "Recommended Products"
        in result.columns
    )


# =============================================================
# TOP RECOMMENDATION
# =============================================================

def test_top_recommendation():

    builder = (
        UnifiedCustomerDatasetBuilder()
    )

    result = (
        builder.build(
            segmentation_df=
                create_segmentation_data(),

            recommendation_df=
                create_recommendation_data()
        )
    )

    customer = (
        result[
            result[
                "Customer ID"
            ]
            ==
            101
        ]
        .iloc[0]
    )

    assert (
        customer[
            "Top Recommended Stock Code"
        ]
        ==
        "P1"
    )

    assert (
        customer[
            "Top Recommended Product"
        ]
        ==
        "Product 1"
    )

    assert (
        customer[
            "Top Recommendation Score"
        ]
        ==
        0.95
    )


# =============================================================
# RECOMMENDATION LIST
# =============================================================

def test_recommendation_list():

    builder = (
        UnifiedCustomerDatasetBuilder()
    )

    result = (
        builder.build(
            segmentation_df=
                create_segmentation_data(),

            recommendation_df=
                create_recommendation_data()
        )
    )

    customer = (
        result[
            result[
                "Customer ID"
            ]
            ==
            101
        ]
        .iloc[0]
    )

    assert (
        customer[
            "Recommended Stock Codes"
        ]
        ==
        "P1 | P2"
    )

    assert (
        customer[
            "Recommended Products"
        ]
        ==
        "Product 1 | Product 2"
    )


# =============================================================
# COVERAGE
# =============================================================

def test_unified_customer_coverage():

    builder = (
        UnifiedCustomerDatasetBuilder()
    )

    df = pd.DataFrame(
        {
            "Customer ID": [
                101,
                102,
                103
            ],

            "Customer Segment": [
                "A",
                "B",
                "C"
            ],

            "Churn Probability": [
                0.1,
                0.8,
                None
            ],

            "Predicted 90-Day Revenue": [
                1000,
                500,
                250
            ]
        }
    )

    coverage = (
        builder.analyze_coverage(
            df
        )
    )

    churn_row = (
        coverage[
            coverage[
                "Module Field"
            ]
            ==
            "Churn Probability"
        ]
        .iloc[0]
    )

    assert (
        churn_row[
            "Available Customers"
        ]
        ==
        2
    )

    assert (
        churn_row[
            "Missing Customers"
        ]
        ==
        1
    )

    assert round(
        churn_row[
            "Coverage Percentage"
        ],
        2
    ) == 66.67


# =============================================================
# QUALITY
# =============================================================

def test_unified_customer_quality():

    builder = (
        UnifiedCustomerDatasetBuilder()
    )

    df = pd.DataFrame(
        {
            "Customer ID": [
                101,
                102,
                103
            ],

            "Customer Segment": [
                "A",
                "B",
                "C"
            ]
        }
    )

    quality = (
        builder.analyze_quality(
            df
        )
    )

    assert (
        quality[
            "row_count"
        ]
        ==
        3
    )

    assert (
        quality[
            "unique_customers"
        ]
        ==
        3
    )

    assert (
        quality[
            "duplicate_customers"
        ]
        ==
        0
    )

    assert (
        quality[
            "missing_customer_ids"
        ]
        ==
        0
    )

    assert (
        quality[
            "one_row_per_customer"
        ]
        is True
    )


# =============================================================
# LEFT JOIN BEHAVIOR
# =============================================================

def test_customer_preserved_when_module_missing():

    segmentation = (
        create_segmentation_data()
    )

    churn = pd.DataFrame(
        {
            "Customer ID": [
                101,
                102
            ],

            "Churn Probability": [
                0.10,
                0.80
            ]
        }
    )

    builder = (
        UnifiedCustomerDatasetBuilder()
    )

    result = (
        builder.build(
            segmentation_df=
                segmentation,

            churn_df=
                churn
        )
    )

    # Customer 103 should still remain
    # because segmentation is master table.

    assert (
        103
        in
        result[
            "Customer ID"
        ].values
    )

    customer_103 = (
        result[
            result[
                "Customer ID"
            ]
            ==
            103
        ]
        .iloc[0]
    )

    assert pd.isna(
        customer_103[
            "Churn Probability"
        ]
    )