import pandas as pd

from src.feature_engineering.segmentation_features import (
    SegmentationFeatureEngineering
)


def test_create_segmentation_features():

    df = pd.DataFrame(
        {
            "CustomerID": [
                101,
                102,
                103
            ],

            "Recency": [
                10,
                50,
                100
            ],

            "Frequency": [
                8,
                4,
                1
            ],

            "Monetary": [
                2000,
                800,
                100
            ],

            "TotalItems": [
                200,
                80,
                10
            ],

            "FirstPurchaseDate": [
                "2021-01-01",
                "2021-02-01",
                "2021-03-01"
            ],

            "LastPurchaseDate": [
                "2021-12-01",
                "2021-10-01",
                "2021-04-01"
            ],

            "AverageOrderValue": [
                250,
                200,
                100
            ],

            "Tenure": [
                334,
                242,
                31
            ],

            "Churn": [
                0,
                0,
                1
            ]
        }
    )

    feature_engineering = (
        SegmentationFeatureEngineering()
    )

    result = (
        feature_engineering
        .create_segmentation_features(
            df
        )
    )

    expected_columns = [
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "TotalItems",
        "AverageOrderValue",
        "Tenure"
    ]

    assert result is not None

    assert (
        result.columns.tolist()
        ==
        expected_columns
    )

    assert (
        len(result)
        == 3
    )

    assert (
        result["CustomerID"]
        .is_unique
    )

    assert (
        result
        .isnull()
        .sum()
        .sum()
        == 0
    )

    assert (
        "Churn"
        not in result.columns
    )

    assert (
        "FirstPurchaseDate"
        not in result.columns
    )

    assert (
        "LastPurchaseDate"
        not in result.columns
    )