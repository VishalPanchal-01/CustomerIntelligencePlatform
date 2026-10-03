import numpy as np
import pandas as pd

from src.recommendation.hybrid_recommender import (
    HybridRecommender
)


def create_hybrid_test_data():

    return pd.DataFrame(
        {
            "CustomerID": [
                101,
                101,
                102,
                102,
                103,
                103,
                104,
                104,
                105,
                105
            ],

            "StockCode": [
                "P1",
                "P2",
                "P1",
                "P3",
                "P2",
                "P3",
                "P3",
                "P4",
                "P4",
                "P5"
            ],

            "Description": [
                "Product 1",
                "Product 2",
                "Product 1",
                "Product 3",
                "Product 2",
                "Product 3",
                "Product 3",
                "Product 4",
                "Product 4",
                "Product 5"
            ],

            "PurchaseCount": [
                3,
                2,
                3,
                2,
                2,
                3,
                2,
                1,
                2,
                1
            ],

            "TotalQuantity": [
                10,
                5,
                9,
                6,
                5,
                8,
                6,
                3,
                5,
                2
            ],

            "TotalSpend": [
                100,
                50,
                90,
                60,
                50,
                80,
                60,
                30,
                50,
                20
            ],

            "InteractionStrength": [
                4.0,
                3.0,
                3.8,
                3.1,
                3.0,
                3.7,
                3.2,
                2.0,
                2.8,
                1.8
            ]
        }
    )


def test_hybrid_fit():

    interactions = (
        create_hybrid_test_data()
    )

    recommender = (
        HybridRecommender(
            item_cf_weight=0.4,
            matrix_factorization_weight=0.4,
            popularity_weight=0.2,
            n_components=2,
            random_state=42
        )
    )

    recommender.fit(
        interactions
    )

    assert (
        recommender.is_fitted
        is True
    )

    assert (
        recommender.popularity_model
        is not None
    )

    assert (
        recommender.item_cf_model
        is not None
    )

    assert (
        recommender.matrix_factorization_model
        is not None
    )

    assert np.isclose(
        recommender.item_cf_weight
        +
        recommender.matrix_factorization_weight
        +
        recommender.popularity_weight,
        1.0
    )


def test_hybrid_recommend():

    interactions = (
        create_hybrid_test_data()
    )

    recommender = (
        HybridRecommender(
            n_components=2,
            random_state=42
        )
    )

    recommender.fit(
        interactions
    )

    recommendations = (
        recommender.recommend(
            customer_id=101,
            top_k=3,
            exclude_already_purchased=False
        )
    )

    assert (
        recommendations
        is not None
    )

    assert (
        len(
            recommendations
        )
        > 0
    )

    expected_columns = [
        "StockCode",
        "Description",
        "HybridScore",
        "ItemCFScore",
        "MatrixFactorizationScore",
        "PopularityScore",
        "RecommendationRank",
        "RecommendationSource"
    ]

    assert (
        recommendations.columns.tolist()
        ==
        expected_columns
    )

    assert (
        recommendations[
            "HybridScore"
        ]
        >= 0
    ).all()

    assert np.isfinite(
        recommendations[
            "HybridScore"
        ]
    ).all()


def test_hybrid_exclude_purchased():

    interactions = (
        create_hybrid_test_data()
    )

    recommender = (
        HybridRecommender(
            n_components=2,
            random_state=42
        )
    )

    recommender.fit(
        interactions
    )

    recommendations = (
        recommender.recommend(
            customer_id=101,
            top_k=5,
            exclude_already_purchased=True
        )
    )

    purchased = {
        "P1",
        "P2"
    }

    recommended = set(
        recommendations[
            "StockCode"
        ]
    )

    assert purchased.isdisjoint(
        recommended
    )


def test_hybrid_cold_start():

    interactions = (
        create_hybrid_test_data()
    )

    recommender = (
        HybridRecommender(
            n_components=2,
            random_state=42
        )
    )

    recommender.fit(
        interactions
    )

    recommendations = (
        recommender.recommend(
            customer_id=999999,
            top_k=3
        )
    )

    assert (
        len(
            recommendations
        )
        == 3
    )

    assert (
        recommendations[
            "RecommendationSource"
        ]
        ==
        "Popularity Fallback"
    ).all()


def test_hybrid_weight_normalization():

    recommender = (
        HybridRecommender(
            item_cf_weight=4,
            matrix_factorization_weight=4,
            popularity_weight=2
        )
    )

    assert np.isclose(
        recommender.item_cf_weight,
        0.4
    )

    assert np.isclose(
        recommender.matrix_factorization_weight,
        0.4
    )

    assert np.isclose(
        recommender.popularity_weight,
        0.2
    )