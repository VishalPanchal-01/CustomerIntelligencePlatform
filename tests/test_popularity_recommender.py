import pandas as pd

from src.recommendation.popularity_recommender import (
    PopularityRecommender
)


def create_interaction_data():

    return pd.DataFrame(
        {
            "CustomerID": [
                101,
                102,
                103,
                101,
                102
            ],

            "StockCode": [
                "P1",
                "P1",
                "P1",
                "P2",
                "P2"
            ],

            "Description": [
                "Product 1",
                "Product 1",
                "Product 1",
                "Product 2",
                "Product 2"
            ],

            "PurchaseCount": [
                3,
                2,
                1,
                1,
                1
            ],

            "TotalQuantity": [
                10,
                8,
                5,
                2,
                2
            ],

            "TotalSpend": [
                100,
                80,
                50,
                20,
                20
            ],

            "InteractionStrength": [
                4.0,
                3.5,
                2.5,
                1.5,
                1.5
            ]
        }
    )


def test_popularity_recommender():

    interactions = (
        create_interaction_data()
    )

    recommender = (
        PopularityRecommender()
    )

    recommender.fit(
        interactions
    )

    assert (
        recommender.product_ranking
        is not None
    )

    assert (
        len(
            recommender.product_ranking
        )
        == 2
    )

    # P1 should be more popular
    assert (
        recommender
        .product_ranking[
            "StockCode"
        ]
        .iloc[0]
        ==
        "P1"
    )

    recommendations = (
        recommender.recommend(
            customer_id=101,
            top_k=2,
            exclude_already_purchased=False
        )
    )

    assert (
        len(
            recommendations
        )
        == 2
    )


def test_popularity_exclude_purchased():

    interactions = (
        create_interaction_data()
    )

    recommender = (
        PopularityRecommender()
    )

    recommender.fit(
        interactions
    )

    # Customer 103 has purchased P1
    # but has not purchased P2.
    recommendations = (
        recommender.recommend(
            customer_id=103,
            top_k=5,
            exclude_already_purchased=True
        )
    )

    assert (
        "P1"
        not in
        recommendations[
            "StockCode"
        ].values
    )

    assert (
        "P2"
        in
        recommendations[
            "StockCode"
        ].values
    )