import numpy as np
import pandas as pd

from src.recommendation.matrix_factorization import (
    MatrixFactorizationRecommender
)


def create_matrix_factorization_data():

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
                104
            ],

            "StockCode": [
                "P1",
                "P2",
                "P1",
                "P3",
                "P2",
                "P3",
                "P3",
                "P4"
            ],

            "Description": [
                "Product 1",
                "Product 2",
                "Product 1",
                "Product 3",
                "Product 2",
                "Product 3",
                "Product 3",
                "Product 4"
            ],

            "InteractionStrength": [
                3.0,
                2.0,
                3.2,
                2.5,
                2.8,
                3.0,
                2.7,
                2.2
            ]
        }
    )


def test_matrix_factorization_fit():

    interactions = (
        create_matrix_factorization_data()
    )

    recommender = (
        MatrixFactorizationRecommender(
            n_components=2,
            random_state=42
        )
    )

    recommender.fit(
        interactions
    )

    assert (
        recommender.model
        is not None
    )

    assert (
        recommender.customer_item_matrix
        is not None
    )

    assert (
        recommender.user_factors
        is not None
    )

    assert (
        recommender.item_factors
        is not None
    )

    assert (
        recommender.customer_item_matrix.shape
        ==
        (
            4,
            4
        )
    )

    assert (
        recommender.user_factors.shape
        ==
        (
            4,
            2
        )
    )

    assert (
        recommender.item_factors.shape
        ==
        (
            2,
            4
        )
    )


def test_matrix_factorization_recommend():

    interactions = (
        create_matrix_factorization_data()
    )

    recommender = (
        MatrixFactorizationRecommender(
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
            top_k=4,
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

    assert (
        "StockCode"
        in recommendations.columns
    )

    assert (
        "Description"
        in recommendations.columns
    )

    assert (
        "RecommendationScore"
        in recommendations.columns
    )

    assert (
        "RecommendationRank"
        in recommendations.columns
    )

    assert np.isfinite(
        recommendations[
            "RecommendationScore"
        ]
    ).all()


def test_matrix_factorization_exclude_purchased():

    interactions = (
        create_matrix_factorization_data()
    )

    recommender = (
        MatrixFactorizationRecommender(
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
            top_k=4,
            exclude_already_purchased=True
        )
    )

    purchased_products = {
        "P1",
        "P2"
    }

    recommended_products = set(
        recommendations[
            "StockCode"
        ]
    )

    assert (
        purchased_products
        .isdisjoint(
            recommended_products
        )
    )


def test_matrix_factorization_unknown_customer():

    interactions = (
        create_matrix_factorization_data()
    )

    recommender = (
        MatrixFactorizationRecommender(
            n_components=2,
            random_state=42
        )
    )

    recommender.fit(
        interactions
    )

    recommendations = (
        recommender.recommend(
            customer_id=999,
            top_k=5
        )
    )

    assert (
        recommendations.empty
    )


def test_matrix_factorization_component_reduction():

    interactions = (
        create_matrix_factorization_data()
    )

    recommender = (
        MatrixFactorizationRecommender(
            n_components=50,
            random_state=42
        )
    )

    recommender.fit(
        interactions
    )

    # Matrix is 4 x 4.
    # Therefore at most 3 components
    # are used by our implementation.

    assert (
        recommender.actual_n_components
        ==
        3
    )