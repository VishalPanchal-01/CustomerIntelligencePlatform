import pandas as pd

from src.feature_engineering.clv_dataset import (
    CLVDatasetBuilder
)


def test_clv_dataset_builder():

    df = pd.DataFrame(
        {
            "Customer ID": [
                101,
                101,
                101,
                102,
                102,
                103,
                103
            ],

            "Invoice": [
                "A1",
                "A2",
                "A3",
                "B1",
                "B2",
                "C1",
                "C2"
            ],

            "InvoiceDate": pd.to_datetime(
                [
                    "2021-01-01",
                    "2021-02-01",
                    "2021-04-20",
                    "2021-01-10",
                    "2021-02-10",
                    "2021-01-15",
                    "2021-05-01"
                ]
            ),

            "Quantity": [
                10,
                5,
                4,
                8,
                6,
                3,
                2
            ],

            "Revenue": [
                100,
                50,
                200,
                80,
                60,
                30,
                120
            ]
        }
    )

    builder = (
        CLVDatasetBuilder(
            prediction_days=60
        )
    )

    result = (
        builder.build_dataset(
            df
        )
    )

    assert result is not None

    assert (
        "FutureRevenue"
        in result.columns
    )

    assert (
        "Recency"
        in result.columns
    )

    assert (
        "Frequency"
        in result.columns
    )

    assert (
        "Monetary"
        in result.columns
    )

    assert (
        "AverageOrderValue"
        in result.columns
    )

    assert (
        "Tenure"
        in result.columns
    )

    assert (
        result[
            "FutureRevenue"
        ]
        .isnull()
        .sum()
        == 0
    )

    assert (
        result[
            "FutureRevenue"
        ]
        >= 0
    ).all()

    assert (
        result[
            "Customer ID"
        ]
        .is_unique
    )