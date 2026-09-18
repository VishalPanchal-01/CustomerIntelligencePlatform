import os

import pandas as pd

from src.visualization.segmentation_visualization import (
    SegmentationVisualization
)


def test_segmentation_visualization(
    tmp_path
):

    clustered_df = pd.DataFrame(
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

    scaled_features = pd.DataFrame(
        {
            "Recency": [
                -1.2,
                -1.0,
                1.1,
                1.3,
                -0.2,
                0.0
            ],

            "Frequency": [
                1.3,
                1.1,
                -1.0,
                -1.2,
                0.2,
                0.0
            ],

            "Monetary": [
                1.4,
                1.2,
                -1.1,
                -1.3,
                0.2,
                0.0
            ],

            "TotalItems": [
                1.3,
                1.1,
                -1.0,
                -1.2,
                0.1,
                0.0
            ],

            "AverageOrderValue": [
                1.0,
                1.0,
                -1.0,
                -0.8,
                0.2,
                0.0
            ],

            "Tenure": [
                1.4,
                1.2,
                -1.1,
                -1.3,
                0.1,
                0.0
            ]
        }
    )

    clusters = pd.Series(
        [
            0,
            0,
            1,
            1,
            2,
            2
        ],
        name="Cluster"
    )

    visualizer = (
        SegmentationVisualization(
            output_directory=
                str(tmp_path)
        )
    )

    distribution_path = (
        visualizer
        .plot_cluster_distribution(
            clustered_df
        )
    )

    profile_path = (
        visualizer
        .plot_cluster_feature_profile(
            clustered_df
        )
    )

    pca_path = (
        visualizer
        .plot_pca_clusters(
            scaled_features,
            clusters
        )
    )

    assert os.path.exists(
        distribution_path
    )

    assert os.path.exists(
        profile_path
    )

    assert os.path.exists(
        pca_path
    )