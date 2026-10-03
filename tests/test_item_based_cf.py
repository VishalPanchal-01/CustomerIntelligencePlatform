import pandas as pd

from src.recommendation.item_based_cf import (
    ItemBasedCollaborativeRecommender
)


def create_item_cf_data():

    return pd.DataFrame(
        {
            "CustomerID": [
                101,
                101,
                102,
                102,
                103,
                103
            ],

            "StockCode": [
                "P1",
                "P2",
                "P1",
                "P3",
                "P2",
                "P3"
            ],

            "Description": [
                "Product 1",
                "Product 2",
                "Product 1",
                "Product 3",
                "Product 2",
                "Product 3"
            ],

            "InteractionStrength": [
                3.0,
                2.0,
                3.0,
                2.5,
                2.5,
                3.0
            ]
        }
    )


def test_item_based_cf_fit():

    interactions = (
        create_item_cf_data()
    )

    recommender = (
        ItemBasedCollaborativeRecommender()
    )

    recommender.fit(
        interactions
    )

    assert (
        recommender.customer_item_matrix
        is not None
    )

    assert (
        recommender.item_similarity_matrix
        is not None
    )

    assert (
        len(
            recommender.customer_ids
        )
        == 3
    )

    assert (
        len(
            recommender.product_ids
        )
        == 3
    )

    assert (
        recommender.customer_item_matrix.shape
        ==
        (
            3,
            3
        )
    )


def test_item_based_cf_recommend():

    interactions = (
        create_item_cf_data()
    )

    recommender = (
        ItemBasedCollaborativeRecommender()
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


def test_item_based_cf_exclude_purchased():

    interactions = (
        create_item_cf_data()
    )

    recommender = (
        ItemBasedCollaborativeRecommender()
    )

    recommender.fit(
        interactions
    )

    recommendations = (
        recommender.recommend(
            customer_id=101,
            top_k=3,
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

    assert (
        purchased
        .isdisjoint(
            recommended
        )
    )


def test_item_based_cf_unknown_customer():

    interactions = (
        create_item_cf_data()
    )

    recommender = (
        ItemBasedCollaborativeRecommender()
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