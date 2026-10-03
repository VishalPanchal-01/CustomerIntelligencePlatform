import os

import pandas as pd

from src.recommendation.hybrid_recommender import (
    HybridRecommender
)

from src.utils.recommendation_persistence import (
    RecommendationPersistence
)


def create_recommendation_test_data():

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

            "PurchaseCount": [
                3,
                2,
                3,
                2,
                2,
                3,
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
                3
            ],

            "TotalSpend": [
                100,
                50,
                90,
                60,
                50,
                80,
                60,
                30
            ],

            "InteractionStrength": [
                4.0,
                3.0,
                3.8,
                3.1,
                3.0,
                3.7,
                3.2,
                2.0
            ]
        }
    )


def test_recommendation_persistence(
    tmp_path
):

    interactions = (
        create_recommendation_test_data()
    )

    model = (
        HybridRecommender(
            n_components=2,
            random_state=42
        )
    )

    model.fit(
        interactions
    )

    file_path = os.path.join(
        tmp_path,
        "recommender.pkl"
    )

    persistence = (
        RecommendationPersistence()
    )

    persistence.save_model(
        model,
        file_path
    )

    assert os.path.exists(
        file_path
    )

    loaded_model = (
        persistence.load_model(
            file_path
        )
    )

    original = (
        model.recommend(
            customer_id=101,
            top_k=3,
            exclude_already_purchased=False
        )
    )

    loaded = (
        loaded_model.recommend(
            customer_id=101,
            top_k=3,
            exclude_already_purchased=False
        )
    )

    assert (
        original[
            "StockCode"
        ]
        .tolist()
        ==
        loaded[
            "StockCode"
        ]
        .tolist()
    )