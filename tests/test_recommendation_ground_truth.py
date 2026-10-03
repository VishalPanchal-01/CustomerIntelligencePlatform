import pandas as pd

from src.evaluation.recommendation_ground_truth import (
    RecommendationGroundTruthBuilder
)


def test_recommendation_ground_truth():

    train = pd.DataFrame(
        {
            "CustomerID": [
                101,
                102,
                103
            ],

            "StockCode": [
                "P1",
                "P2",
                "P3"
            ]
        }
    )

    test = pd.DataFrame(
        {
            "CustomerID": [
                101,
                101
            ],

            "StockCode": [
                "P2",
                "P99"
            ],

            "Description": [
                "Known Product",
                "New Product"
            ],

            "Invoice": [
                "A3",
                "A3"
            ],

            "InvoiceDate": pd.to_datetime(
                [
                    "2025-03-01",
                    "2025-03-01"
                ]
            )
        }
    )

    builder = (
        RecommendationGroundTruthBuilder()
    )

    (
        ground_truth,
        evaluable_ground_truth
    ) = builder.build_ground_truth(
        test_transactions=
            test,

        train_transactions=
            train
    )

    assert (
        len(
            ground_truth
        )
        == 2
    )

    # P2 exists in training
    known_product = (
        ground_truth[
            ground_truth[
                "StockCode"
            ]
            ==
            "P2"
        ]
        .iloc[0]
    )

    assert (
        bool(
            known_product[
                "IsEvaluableItem"
            ]
        )
        is True
    )

    # P99 is completely new
    cold_product = (
        ground_truth[
            ground_truth[
                "StockCode"
            ]
            ==
            "P99"
        ]
        .iloc[0]
    )

    assert (
        bool(
            cold_product[
                "IsEvaluableItem"
            ]
        )
        is False
    )

    assert (
        len(
            evaluable_ground_truth
        )
        == 1
    )

    assert (
        evaluable_ground_truth[
            "StockCode"
        ]
        .iloc[0]
        ==
        "P2"
    )