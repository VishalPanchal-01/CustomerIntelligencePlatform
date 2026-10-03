import pandas as pd

from src.preprocessing.recommendation_cleaning import (
    RecommendationDataCleaning
)


def test_recommendation_cleaning():

    df = pd.DataFrame(
        {
            "CustomerID": [
                101,
                101,
                102,
                103,
                104
            ],

            "Invoice": [
                "A1",
                "C2",
                "A3",
                "A4",
                "A5"
            ],

            "StockCode": [
                "P1",
                "P2",
                "P3",
                "P4",
                "P5"
            ],

            "Description": [
                "Product 1",
                "Product 2",
                "Product 3",
                "Product 4",
                "Product 5"
            ],

            "Quantity": [
                2,
                -1,
                3,
                0,
                5
            ],

            "InvoiceDate": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-04",
                    "2025-01-05"
                ]
            ),

            "UnitPrice": [
                10,
                20,
                15,
                30,
                5
            ]
        }
    )

    cleaner = (
        RecommendationDataCleaning()
    )

    result = (
        cleaner.clean_purchase_transactions(
            df
        )
    )

    assert result is not None

    assert (
        len(result)
        == 3
    )

    assert (
        result[
            "Quantity"
        ]
        > 0
    ).all()

    assert (
        result[
            "UnitPrice"
        ]
        > 0
    ).all()

    assert not (
        result[
            "Invoice"
        ]
        .str.upper()
        .str.startswith(
            "C"
        )
    ).any()

    assert (
        "Revenue"
        in result.columns
    )