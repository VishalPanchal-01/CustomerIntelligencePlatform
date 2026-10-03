import pandas as pd

from src.recommendation.popularity_recommender import (
    PopularityRecommender
)

from src.evaluation.recommendation_metrics import (
    RecommendationMetrics
)


def test_recommendation_metrics():

    interactions = pd.DataFrame(
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

    ground_truth = pd.DataFrame(
        {
            "CustomerID": [
                201,
                202
            ],

            "StockCode": [
                "P1",
                "P2"
            ]
        }
    )

    recommender = (
        PopularityRecommender()
    )

    recommender.fit(
        interactions
    )

    evaluator = (
        RecommendationMetrics()
    )

    result = (
        evaluator.evaluate(
            recommender=
                recommender,

            ground_truth=
                ground_truth,

            catalog_products=
                recommender.catalog_products,

            top_k=
                2,

            exclude_already_purchased=
                False
        )
    )

    metrics = (
        result[
            "metrics"
        ]
    )

    # Both products P1 and P2 are
    # included in Top-2.
    #
    # Each user has one ground truth item.
    #
    # Precision@2 = 1 / 2 = 0.5
    # Recall@2    = 1 / 1 = 1.0
    # HitRate@2   = 1.0

    assert (
        metrics[
            "precision_at_k"
        ]
        ==
        0.5
    )

    assert (
        metrics[
            "recall_at_k"
        ]
        ==
        1.0
    )

    assert (
        metrics[
            "hit_rate_at_k"
        ]
        ==
        1.0
    )

    assert (
        metrics[
            "catalog_coverage"
        ]
        ==
        1.0
    )

    assert (
        len(
            result[
                "user_results"
            ]
        )
        == 2
    )