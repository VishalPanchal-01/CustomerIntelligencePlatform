import json
import os

import pandas as pd

from src.recommendation.hybrid_recommender import (
    HybridRecommender
)

from src.utils.recommendation_persistence import (
    RecommendationPersistence
)

from src.prediction.recommendation_predictor import (
    RecommendationPredictor
)


def create_test_data():

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


def create_predictor_files(
    tmp_path
):

    interactions = (
        create_test_data()
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

    model_path = os.path.join(
        tmp_path,
        "recommender.pkl"
    )

    metadata_path = os.path.join(
        tmp_path,
        "recommender_metadata.json"
    )

    persistence = (
        RecommendationPersistence()
    )

    persistence.save_model(
        model,
        model_path
    )

    metadata = {
        "selected_model":
            "Hybrid Recommender",

        "prediction": {
            "default_top_k":
                10
        }
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file
        )

    return (
        model_path,
        metadata_path
    )


def test_recommendation_predictor_known_customer(
    tmp_path
):

    (
        model_path,
        metadata_path
    ) = create_predictor_files(
        tmp_path
    )

    predictor = (
        RecommendationPredictor(
            model_path=
                model_path,

            metadata_path=
                metadata_path
        )
    )

    result = (
        predictor.recommend(
            customer_id=101,
            top_k=3,
            mode="next_purchase"
        )
    )

    assert (
        result[
            "customer_id"
        ]
        ==
        101
    )

    assert (
        result[
            "known_customer"
        ]
        is True
    )

    assert (
        result[
            "mode"
        ]
        ==
        "next_purchase"
    )

    assert (
        result[
            "recommendation_count"
        ]
        > 0
    )

    assert (
        len(
            result[
                "recommendations"
            ]
        )
        <= 3
    )


def test_recommendation_predictor_discovery_mode(
    tmp_path
):

    (
        model_path,
        metadata_path
    ) = create_predictor_files(
        tmp_path
    )

    predictor = (
        RecommendationPredictor(
            model_path=
                model_path,

            metadata_path=
                metadata_path
        )
    )

    result = (
        predictor.recommend(
            customer_id=101,
            top_k=5,
            mode="discovery"
        )
    )

    purchased = {
        "P1",
        "P2"
    }

    recommended = {
        item[
            "StockCode"
        ]
        for item
        in result[
            "recommendations"
        ]
    }

    assert purchased.isdisjoint(
        recommended
    )

    assert (
        result[
            "exclude_already_purchased"
        ]
        is True
    )


def test_recommendation_predictor_cold_start(
    tmp_path
):

    (
        model_path,
        metadata_path
    ) = create_predictor_files(
        tmp_path
    )

    predictor = (
        RecommendationPredictor(
            model_path=
                model_path,

            metadata_path=
                metadata_path
        )
    )

    result = (
        predictor.recommend(
            customer_id=999999,
            top_k=3,
            mode="next_purchase"
        )
    )

    assert (
        result[
            "known_customer"
        ]
        is False
    )

    assert (
        result[
            "recommendation_count"
        ]
        == 3
    )

    assert (
        len(
            result[
                "recommendations"
            ]
        )
        ==
        3
    )


def test_recommendation_predictor_batch(
    tmp_path
):

    (
        model_path,
        metadata_path
    ) = create_predictor_files(
        tmp_path
    )

    predictor = (
        RecommendationPredictor(
            model_path=
                model_path,

            metadata_path=
                metadata_path
        )
    )

    result = (
        predictor.recommend_batch(
            customer_ids=[
                101,
                102
            ],

            top_k=2,

            mode="next_purchase"
        )
    )

    assert (
        result is not None
    )

    assert (
        "CustomerID"
        in result.columns
    )

    assert (
        "StockCode"
        in result.columns
    )

    assert (
        "Score"
        in result.columns
    )

    assert (
        "Rank"
        in result.columns
    )

    assert set(
        result[
            "CustomerID"
        ]
    ) == {
        101,
        102
    }