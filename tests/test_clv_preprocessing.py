import numpy as np
import pandas as pd

from src.preprocessing.clv_preprocessing import (
    CLVPreprocessor
)


def test_clv_preprocessing():

    df = pd.DataFrame(
        {
            "CustomerID": [
                101,
                102,
                103,
                104,
                105
            ],

            "Recency": [
                10,
                20,
                50,
                100,
                150
            ],

            "Frequency": [
                10,
                8,
                5,
                2,
                1
            ],

            "Monetary": [
                5000,
                3500,
                1500,
                500,
                100
            ],

            "TotalItems": [
                500,
                350,
                150,
                50,
                10
            ],

            "AverageOrderValue": [
                500,
                437.5,
                300,
                250,
                100
            ],

            "Tenure": [
                400,
                350,
                250,
                100,
                20
            ],

            "FutureRevenue": [
                2000,
                1500,
                500,
                0,
                0
            ]
        }
    )

    preprocessor = (
        CLVPreprocessor()
    )

    (
        X,
        y_raw,
        y_log
    ) = preprocessor.prepare_features(
        df
    )

    expected_features = [
        "Recency",
        "Frequency",
        "Monetary",
        "TotalItems",
        "AverageOrderValue",
        "Tenure"
    ]

    assert (
        X.columns.tolist()
        ==
        expected_features
    )

    assert (
        len(X)
        == 5
    )

    assert (
        len(y_raw)
        == 5
    )

    assert (
        len(y_log)
        == 5
    )

    assert (
        X.isnull()
        .sum()
        .sum()
        == 0
    )

    assert (
        y_raw.isnull()
        .sum()
        == 0
    )

    assert (
        y_log.isnull()
        .sum()
        == 0
    )

    assert (
        y_raw >= 0
    ).all()

    assert (
        y_log >= 0
    ).all()

    expected_log = (
        np.log1p(
            y_raw
        )
    )

    assert np.allclose(
        y_log,
        expected_log
    )