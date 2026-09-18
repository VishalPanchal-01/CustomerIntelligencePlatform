import pandas as pd

from src.analysis.segment_interpretation import (
    SegmentInterpreter
)


def test_segment_interpretation():

    df = pd.DataFrame(
        {
            "CustomerID": [
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108
            ],

            "Recency": [
                10,
                15,
                180,
                200,
                20,
                25,
                80,
                90
            ],

            "Frequency": [
                12,
                10,
                1,
                2,
                2,
                1,
                5,
                4
            ],

            "Monetary": [
                5000,
                4500,
                200,
                300,
                1000,
                900,
                1800,
                1700
            ],

            "TotalItems": [
                500,
                450,
                20,
                30,
                100,
                90,
                180,
                170
            ],

            "AverageOrderValue": [
                416,
                450,
                200,
                150,
                500,
                900,
                360,
                425
            ],

            "Tenure": [
                400,
                380,
                50,
                40,
                30,
                25,
                250,
                230
            ],

            "Cluster": [
                0,
                0,
                1,
                1,
                2,
                2,
                3,
                3
            ]
        }
    )

    interpreter = (
        SegmentInterpreter()
    )

    result = (
        interpreter
        .create_segment_summary(
            df
        )
    )

    assert result is not None

    assert (
        len(result)
        == 4
    )

    assert (
        "SegmentName"
        in result.columns
    )

    assert (
        "Recommendation"
        in result.columns
    )

    assert (
        "RecentPurchase"
        in result.columns
    )

    assert (
        "HighFrequency"
        in result.columns
    )

    assert (
        "HighMonetary"
        in result.columns
    )

    assert (
        "LongTenure"
        in result.columns
    )

    assert (
        result[
            "SegmentName"
        ]
        .notnull()
        .all()
    )

    assert (
        result[
            "Recommendation"
        ]
        .notnull()
        .all()
    )

    assert (
        result[
            "CustomerCount"
        ]
        .sum()
        ==
        8
    )

    assert round(
        result[
            "CustomerPercentage"
        ].sum(),
        2
    ) == 100.00