import numpy as np
import pandas as pd

from src.preprocessing.segmentation_preprocessing import (
    SegmentationPreprocessor
)


def test_segmentation_preprocessing():

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
                30,
                50,
                100,
                200
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
                3000,
                1500,
                500,
                100
            ],

            "TotalItems": [
                500,
                300,
                150,
                50,
                10
            ],

            "AverageOrderValue": [
                500,
                375,
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
            ]
        }
    )

    preprocessor = (
        SegmentationPreprocessor()
    )

    (
        customer_ids,
        scaled_features
    ) = preprocessor.prepare_features(
        df
    )

    assert customer_ids is not None

    assert scaled_features is not None

    assert len(
        customer_ids
    ) == 5

    assert (
        scaled_features.shape
        ==
        (5, 6)
    )

    assert (
        "CustomerID"
        not in scaled_features.columns
    )

    assert (
        scaled_features
        .isnull()
        .sum()
        .sum()
        == 0
    )

    assert np.isfinite(
        scaled_features.to_numpy()
    ).all()

    means = (
        scaled_features
        .mean()
        .abs()
    )

    assert (
        means < 1e-10
    ).all()