import pandas as pd

from dashboard.customer_explorer import (
    CustomerExplorer
)


# =============================================================
# TEST DATA
# =============================================================

def create_customer_data():

    return pd.DataFrame(
        {
            "Customer ID": [
                101,
                102,
                103,
                104
            ],

            "Customer Segment": [
                "VIP",
                "At Risk",
                "Regular",
                "VIP"
            ],

            "Recency": [
                10,
                120,
                50,
                5
            ],

            "Frequency": [
                15,
                2,
                6,
                20
            ],

            "Monetary": [
                5000,
                500,
                1500,
                7000
            ],

            "TotalItems": [
                300,
                20,
                90,
                500
            ],

            "AverageOrderValue": [
                333.33,
                250.00,
                250.00,
                350.00
            ],

            "Tenure": [
                500,
                200,
                300,
                600
            ],

            "Churn Probability": [
                0.85,
                0.80,
                0.40,
                0.10
            ],

            "Churn Risk": [
                "High",
                "High",
                "Medium",
                "Low"
            ],

            "Predicted 90-Day Revenue": [
                3000,
                300,
                800,
                4000
            ],

            "CLV Value Band": [
                "High",
                "Low",
                "Medium",
                "High"
            ],

            "Top Recommended Stock Code": [
                "P1",
                "P2",
                "P3",
                "P4"
            ],

            "Top Recommended Product": [
                "Product A",
                "Product B",
                "Product C",
                "Product D"
            ],

            "Top Recommendation Score": [
                0.95,
                0.80,
                0.70,
                0.92
            ],

            "Recommendation Source": [
                "Hybrid Recommender",
                "Hybrid Recommender",
                "Hybrid Recommender",
                "Hybrid Recommender"
            ],

            "Recommended Stock Codes": [
                "P1 | P5 | P6",
                "P2 | P7 | P8",
                "P3 | P9 | P10",
                "P4 | P11 | P12"
            ],

            "Recommended Products": [
                (
                    "Product A | "
                    "Product E | "
                    "Product F"
                ),

                (
                    "Product B | "
                    "Product G | "
                    "Product H"
                ),

                (
                    "Product C | "
                    "Product I | "
                    "Product J"
                ),

                (
                    "Product D | "
                    "Product K | "
                    "Product L"
                )
            ]
        }
    )


# =============================================================
# GET CUSTOMER
# =============================================================

def test_get_customer():

    explorer = (
        CustomerExplorer()
    )

    customer = (
        explorer.get_customer(
            create_customer_data(),
            customer_id=101
        )
    )

    assert customer is not None

    assert (
        customer[
            "Customer ID"
        ]
        ==
        101
    )

    assert (
        customer[
            "Customer Segment"
        ]
        ==
        "VIP"
    )


# =============================================================
# BUSINESS PRIORITY
# =============================================================

def test_customer_priority():

    explorer = (
        CustomerExplorer()
    )

    result = (
        explorer.customer_priority(
            create_customer_data(),
            customer_id=101
        )
    )

    # Customer 101:
    # High CLV + High churn
    # = Critical

    assert (
        result[
            "priority"
        ]
        ==
        "Critical"
    )

    assert (
        result[
            "business_action"
        ]
        is not None
    )


# =============================================================
# RETENTION PRIORITY SCORE
# =============================================================

def test_retention_priority_score():

    explorer = (
        CustomerExplorer()
    )

    score = (
        explorer.retention_priority_score(
            create_customer_data(),
            customer_id=101
        )
    )

    assert score is not None

    assert (
        score
        >=
        0
    )

    assert (
        score
        <=
        1
    )


# =============================================================
# RETENTION RANK
# =============================================================

def test_retention_rank():

    explorer = (
        CustomerExplorer()
    )

    rank = (
        explorer.retention_rank(
            create_customer_data(),
            customer_id=101
        )
    )

    assert rank is not None

    assert (
        rank
        >=
        1
    )


# =============================================================
# BEHAVIOR PERCENTILES
# =============================================================

def test_behavior_percentiles():

    explorer = (
        CustomerExplorer()
    )

    result = (
        explorer.behavior_percentiles(
            create_customer_data(),
            customer_id=101
        )
    )

    assert not result.empty

    assert (
        "Feature"
        in result.columns
    )

    assert (
        "Customer Value"
        in result.columns
    )

    assert (
        "Population Average"
        in result.columns
    )

    assert (
        "Percentile"
        in result.columns
    )

    assert (
        result[
            "Percentile"
        ]
        .between(
            0,
            100
        )
        .all()
    )


# =============================================================
# NORMALIZED PROFILE
# =============================================================

def test_normalized_behavior_profile():

    explorer = (
        CustomerExplorer()
    )

    result = (
        explorer.normalized_behavior_profile(
            create_customer_data(),
            customer_id=101
        )
    )

    assert not result.empty

    assert (
        "Recency Engagement"
        in result[
            "Feature"
        ].values
    )


# =============================================================
# RECOMMENDATIONS
# =============================================================

def test_customer_recommendations():

    explorer = (
        CustomerExplorer()
    )

    result = (
        explorer.recommendations(
            create_customer_data(),
            customer_id=101
        )
    )

    assert (
        len(result)
        ==
        3
    )

    assert (
        result[
            "Stock Code"
        ]
        .tolist()
        ==
        [
            "P1",
            "P5",
            "P6"
        ]
    )


# =============================================================
# CUSTOMER REPORT
# =============================================================

def test_customer_report():

    explorer = (
        CustomerExplorer()
    )

    result = (
        explorer.build_customer_report(
            create_customer_data(),
            customer_id=101
        )
    )

    assert not result.empty

    assert (
        "Category"
        in result.columns
    )

    assert (
        "Metric"
        in result.columns
    )

    assert (
        "Value"
        in result.columns
    )

    metrics = (
        result[
            "Metric"
        ]
        .tolist()
    )

    assert (
        "Customer ID"
        in metrics
    )

    assert (
        "Churn Probability"
        in metrics
    )

    assert (
        "Predicted 90-Day Revenue"
        in metrics
    )

    assert (
        "Customer Priority"
        in metrics
    )

    assert (
        "Recommended Business Action"
        in metrics
    )


# =============================================================
# CSV EXPORT
# =============================================================

def test_customer_report_csv():

    explorer = (
        CustomerExplorer()
    )

    csv_data = (
        explorer.customer_report_csv(
            create_customer_data(),
            customer_id=101
        )
    )

    assert isinstance(
        csv_data,
        bytes
    )

    assert (
        len(
            csv_data
        )
        >
        0
    )

    text = (
        csv_data
        .decode(
            "utf-8"
        )
    )

    assert (
        "Customer ID"
        in text
    )

    assert (
        "Recommended Business Action"
        in text
    )