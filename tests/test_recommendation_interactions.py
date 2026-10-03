import numpy as np
import pandas as pd

from src.feature_engineering.recommendation_interactions import (
    RecommendationInteractionBuilder
)


def test_recommendation_interactions():

    df = pd.DataFrame(
        {
            "CustomerID": [
                101,
                101,
                101,
                102
            ],

            "Invoice": [
                "A1",
                "A2",
                "A2",
                "B1"
            ],

            "StockCode": [
                "P1",
                "P1",
                "P2",
                "P1"
            ],

            "Description": [
                "Product 1",
                "Product 1",
                "Product 2",
                "Product 1"
            ],

            "Quantity": [
                2,
                3,
                1,
                4
            ],

            "InvoiceDate": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-10",
                    "2025-01-10",
                    "2025-01-05"
                ]
            ),

            "Revenue": [
                20,
                30,
                15,
                40
            ]
        }
    )

    builder = (
        RecommendationInteractionBuilder()
    )

    result = (
        builder.build_interactions(
            df
        )
    )

    assert result is not None

    # Customer 101 has:
    # P1 and P2
    #
    # Customer 102 has:
    # P1
    #
    # Therefore total interactions = 3

    assert (
        len(result)
        == 3
    )

    customer_101_p1 = (
        result[
            (
                result[
                    "CustomerID"
                ]
                == 101
            )
            &
            (
                result[
                    "StockCode"
                ]
                == "P1"
            )
        ]
        .iloc[0]
    )

    assert (
        customer_101_p1[
            "PurchaseCount"
        ]
        == 2
    )

    assert (
        customer_101_p1[
            "TotalQuantity"
        ]
        == 5
    )

    assert (
        customer_101_p1[
            "TotalSpend"
        ]
        == 50
    )

    expected_strength = (
        np.log1p(
            2
        )
        +
        np.log1p(
            5
        )
    )

    assert np.isclose(
        customer_101_p1[
            "InteractionStrength"
        ],
        expected_strength
    )