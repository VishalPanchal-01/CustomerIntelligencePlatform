import pandas as pd

from src.evaluation.segmentation_cluster_analysis import (
    SegmentationClusterAnalysis
)


def test_segmentation_cluster_analysis():

    X = pd.DataFrame(
        {
            "Recency": [
                -1.2,
                -1.1,
                -1.0,
                -0.9,
                0.9,
                1.0,
                1.1,
                1.2,
                2.8,
                2.9,
                3.0,
                3.1
            ],

            "Frequency": [
                1.2,
                1.1,
                1.0,
                0.9,
                -0.8,
                -0.9,
                -1.0,
                -1.1,
                2.7,
                2.8,
                2.9,
                3.0
            ],

            "Monetary": [
                1.1,
                1.0,
                0.9,
                1.2,
                -0.9,
                -1.0,
                -1.1,
                -0.8,
                2.8,
                2.7,
                3.0,
                2.9
            ],

            "TotalItems": [
                1.0,
                1.1,
                1.2,
                0.9,
                -1.0,
                -0.9,
                -1.1,
                -0.8,
                2.7,
                2.9,
                2.8,
                3.0
            ],

            "AverageOrderValue": [
                0.9,
                1.0,
                1.1,
                1.2,
                -1.1,
                -1.0,
                -0.9,
                -0.8,
                3.0,
                2.9,
                2.8,
                2.7
            ],

            "Tenure": [
                1.2,
                1.1,
                1.0,
                0.9,
                -0.8,
                -0.9,
                -1.0,
                -1.1,
                2.7,
                2.8,
                2.9,
                3.0
            ]
        }
    )

    analyzer = (
        SegmentationClusterAnalysis()
    )

    result = (
        analyzer.evaluate_clusters(
            X,
            min_clusters=2,
            max_clusters=4
        )
    )

    assert result is not None

    assert (
        len(result)
        == 3
    )

    assert (
        result["K"].tolist()
        ==
        [2, 3, 4]
    )

    expected_columns = [
        "K",
        "Inertia",
        "SilhouetteScore",
        "SmallestCluster",
        "LargestCluster"
    ]

    assert (
        result.columns.tolist()
        ==
        expected_columns
    )

    assert (
        result[
            "Inertia"
        ]
        > 0
    ).all()

    assert (
        result[
            "SilhouetteScore"
        ]
        >= -1
    ).all()

    assert (
        result[
            "SilhouetteScore"
        ]
        <= 1
    ).all()

    assert (
        result[
            "SmallestCluster"
        ]
        > 0
    ).all()

    assert (
        result[
            "LargestCluster"
        ]
        > 0
    ).all()