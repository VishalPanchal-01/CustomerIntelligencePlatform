import pandas as pd

from src.analysis.segmentation_profile import (
    SegmentationProfiler
)


def test_segmentation_profile():

    df = pd.DataFrame(
        {
            "CustomerID": [
                101,
                102,
                103,
                104,
                105,
                106
            ],

            "Recency": [
                10,
                20,
                100,
                120,
                30,
                40
            ],

            "Frequency": [
                10,
                8,
                2,
                1,
                6,
                5
            ],

            "Monetary": [
                5000,
                4000,
                500,
                300,
                2000,
                1800
            ],

            "TotalItems": [
                500,
                400,
                50,
                30,
                200,
                180
            ],

            "AverageOrderValue": [
                500,
                500,
                250,
                300,
                333,
                360
            ],

            "Tenure": [
                400,
                350,
                100,
                50,
                250,
                220
            ],

            "Cluster": [
                0,
                0,
                1,
                1,
                2,
                2
            ]
        }
    )

    profiler = (
        SegmentationProfiler()
    )

    result = (
        profiler
        .create_cluster_profile(
            df
        )
    )

    assert result is not None

    assert (
        len(result)
        == 3
    )

    assert (
        "Cluster"
        in result.columns
    )

    assert (
        "CustomerCount"
        in result.columns
    )

    assert (
        "CustomerPercentage"
        in result.columns
    )

    assert (
        "Recency_Mean"
        in result.columns
    )

    assert (
        "Frequency_Mean"
        in result.columns
    )

    assert (
        "Monetary_Mean"
        in result.columns
    )

    cluster_zero = (
        result[
            result["Cluster"] == 0
        ]
        .iloc[0]
    )

    assert (
        cluster_zero[
            "CustomerCount"
        ]
        == 2
    )

    assert (
        cluster_zero[
            "Recency_Mean"
        ]
        == 15
    )

    assert (
        cluster_zero[
            "Frequency_Mean"
        ]
        == 9
    )

    assert (
        cluster_zero[
            "Monetary_Mean"
        ]
        == 4500
    )
    