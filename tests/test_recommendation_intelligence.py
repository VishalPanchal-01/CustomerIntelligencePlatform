import pandas as pd

from dashboard.recommendation_intelligence import (
    RecommendationIntelligence
)


# =============================================================
# TEST DATA
# =============================================================

def create_recommendation_data():

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
                "VIP",
                "Regular",
                "Regular",
                "At Risk",
                "At Risk"
            ],

            "Churn Risk": [
                "Low",
                "High",
                "Medium",
                "Low",
                "High",
                "High"
            ],

            "CLV Value Band": [
                "High",
                "High",
                "Medium",
                "Medium",
                "Low",
                "Low"
            ],

            "Top Recommended Stock Code": [
                "P1",
                "P1",
                "P2",
                "P3",
                "P2",
                None
            ],

            "Top Recommended Product": [
                "Product A",
                "Product A",
                "Product B",
                "Product C",
                "Product B",
                None
            ],

            "Top Recommendation Score": [
                0.95,
                0.90,
                0.80,
                0.75,
                0.70,
                None
            ],

            "Recommendation Source": [
                "Hybrid Recommender",
                "Hybrid Recommender",
                "Hybrid Recommender",
                "Hybrid Recommender",
                "Popularity Fallback",
                None
            ],

            "Recommended Stock Codes": [
                "P1 | P2 | P3",
                "P1 | P4 | P5",
                "P2 | P3 | P6",
                "P3 | P7 | P8",
                "P2 | P5 | P9",
                None
            ],

            "Recommended Products": [
                (
                    "Product A | "
                    "Product B | "
                    "Product C"
                ),
                (
                    "Product A | "
                    "Product D | "
                    "Product E"
                ),
                (
                    "Product B | "
                    "Product C | "
                    "Product F"
                ),
                (
                    "Product C | "
                    "Product G | "
                    "Product H"
                ),
                (
                    "Product B | "
                    "Product E | "
                    "Product I"
                ),
                None
            ]
        }
    )


# =============================================================
# METRICS
# =============================================================

def test_recommendation_metrics():

    dashboard = (
        RecommendationIntelligence()
    )

    metrics = (
        dashboard.calculate_metrics(
            create_recommendation_data()
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
            "customers_with_recommendations"
        ]
        ==
        5
    )

    assert (
        metrics[
            "customers_without_recommendations"
        ]
        ==
        1
    )

    assert round(
        metrics[
            "recommendation_coverage"
        ],
        2
    ) == 83.33

    assert (
        metrics[
            "unique_top_products"
        ]
        ==
        3
    )

    assert (
        metrics[
            "unique_recommended_products"
        ]
        ==
        9
    )

    assert (
        metrics[
            "average_recommendations_per_customer"
        ]
        ==
        3.0
    )


# =============================================================
# TOP PRODUCT DISTRIBUTION
# =============================================================

def test_top_product_distribution():

    dashboard = (
        RecommendationIntelligence()
    )

    result = (
        dashboard.top_product_distribution(
            create_recommendation_data()
        )
    )

    product_a = (
        result[
            result[
                "Product"
            ]
            ==
            "Product A"
        ][
            "Recommendations"
        ]
        .iloc[0]
    )

    assert (
        product_a
        ==
        2
    )


# =============================================================
# ALL PRODUCT DISTRIBUTION
# =============================================================

def test_all_product_distribution():

    dashboard = (
        RecommendationIntelligence()
    )

    result = (
        dashboard.all_product_distribution(
            create_recommendation_data()
        )
    )

    assert (
        "Product"
        in result.columns
    )

    assert (
        "Recommendation Count"
        in result.columns
    )

    product_b = (
        result[
            result[
                "Product"
            ]
            ==
            "Product B"
        ][
            "Recommendation Count"
        ]
        .iloc[0]
    )

    assert (
        product_b
        ==
        3
    )


# =============================================================
# SOURCE DISTRIBUTION
# =============================================================

def test_source_distribution():

    dashboard = (
        RecommendationIntelligence()
    )

    result = (
        dashboard.source_distribution(
            create_recommendation_data()
        )
    )

    hybrid = (
        result[
            result[
                "Recommendation Source"
            ]
            ==
            "Hybrid Recommender"
        ][
            "Customers"
        ]
        .iloc[0]
    )

    assert (
        hybrid
        ==
        4
    )


# =============================================================
# PRODUCTS BY SEGMENT
# =============================================================

def test_products_by_segment():

    dashboard = (
        RecommendationIntelligence()
    )

    result = (
        dashboard.products_by_segment(
            create_recommendation_data(),
            top_n_per_segment=2
        )
    )

    assert (
        "Customer Segment"
        in result.columns
    )

    assert (
        "Top Recommended Product"
        in result.columns
    )

    assert (
        "Segment Product Rank"
        in result.columns
    )


# =============================================================
# PRODUCTS BY CLV BAND
# =============================================================

def test_products_by_clv_band():

    dashboard = (
        RecommendationIntelligence()
    )

    result = (
        dashboard.products_by_clv_band(
            create_recommendation_data(),
            top_n_per_band=2
        )
    )

    assert (
        "CLV Value Band"
        in result.columns
    )

    assert (
        "Band Product Rank"
        in result.columns
    )


# =============================================================
# PRODUCTS BY CHURN RISK
# =============================================================

def test_products_by_churn_risk():

    dashboard = (
        RecommendationIntelligence()
    )

    result = (
        dashboard.products_by_churn_risk(
            create_recommendation_data(),
            top_n_per_risk=2
        )
    )

    assert (
        "Churn Risk"
        in result.columns
    )

    assert (
        "Risk Product Rank"
        in result.columns
    )


# =============================================================
# CUSTOMER RECOMMENDATIONS
# =============================================================

def test_customer_recommendations():

    dashboard = (
        RecommendationIntelligence()
    )

    result = (
        dashboard.customer_recommendations(
            create_recommendation_data(),
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
            "Rank"
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
            "Stock Code"
        ]
        .tolist()
        ==
        [
            "P1",
            "P2",
            "P3"
        ]
    )

    assert (
        result[
            "Product"
        ]
        .tolist()
        ==
        [
            "Product A",
            "Product B",
            "Product C"
        ]
    )


# =============================================================
# CUSTOMER WITHOUT RECOMMENDATIONS
# =============================================================

def test_customer_without_recommendations():

    dashboard = (
        RecommendationIntelligence()
    )

    result = (
        dashboard.customer_recommendations(
            create_recommendation_data(),
            customer_id=106
        )
    )

    assert result.empty