import json
import os

import pandas as pd

from src.evaluation.final_segmentation_report import (
    FinalSegmentationReport
)


def test_final_segmentation_report(
    tmp_path
):

    segmented_df = pd.DataFrame(
        {
            "CustomerID": [
                101,
                102,
                103,
                104
            ],

            "Recency": [
                10,
                20,
                100,
                120
            ],

            "Frequency": [
                10,
                8,
                2,
                1
            ],

            "Monetary": [
                5000,
                4000,
                500,
                300
            ],

            "TotalItems": [
                500,
                400,
                50,
                30
            ],

            "AverageOrderValue": [
                500,
                500,
                250,
                300
            ],

            "Tenure": [
                400,
                350,
                100,
                50
            ],

            "Cluster": [
                0,
                0,
                1,
                1
            ],

            "SegmentName": [
                "High Value Loyal",
                "High Value Loyal",
                "Low Engagement",
                "Low Engagement"
            ]
        }
    )

    segment_summary = pd.DataFrame(
        {
            "Cluster": [
                0,
                1
            ],

            "Recency": [
                15.0,
                110.0
            ],

            "Frequency": [
                9.0,
                1.5
            ],

            "Monetary": [
                4500.0,
                400.0
            ],

            "TotalItems": [
                450.0,
                40.0
            ],

            "AverageOrderValue": [
                500.0,
                275.0
            ],

            "Tenure": [
                375.0,
                75.0
            ],

            "CustomerCount": [
                2,
                2
            ],

            "CustomerPercentage": [
                50.0,
                50.0
            ],

            "RecentPurchase": [
                True,
                False
            ],

            "HighFrequency": [
                True,
                False
            ],

            "HighMonetary": [
                True,
                False
            ],

            "HighVolume": [
                True,
                False
            ],

            "HighAOV": [
                True,
                False
            ],

            "LongTenure": [
                True,
                False
            ],

            "SegmentName": [
                "High Value Loyal",
                "Low Engagement"
            ],

            "Recommendation": [
                "Retention strategy",
                "Reactivation strategy"
            ]
        }
    )

    feature_names = [
        "Recency",
        "Frequency",
        "Monetary",
        "TotalItems",
        "AverageOrderValue",
        "Tenure"
    ]

    visualization_paths = {
        "cluster_distribution":
            "cluster_distribution.png",

        "cluster_feature_profile":
            "cluster_feature_profile.png",

        "pca_visualization":
            "customer_segments_pca.png"
    }

    output_path = os.path.join(
        tmp_path,
        "segmentation_report.json"
    )

    generator = (
        FinalSegmentationReport()
    )

    report = (
        generator
        .generate_report(
            segmented_df=
                segmented_df,

            segment_summary=
                segment_summary,

            silhouette_score_value=
                0.52,

            inertia=
                100.5,

            selected_k=
                2,

            feature_names=
                feature_names,

            preprocessor_path=
                "models/segmentation/"
                "segmentation_preprocessor.pkl",

            model_path=
                "models/segmentation/"
                "kmeans_model.pkl",

            visualization_paths=
                visualization_paths,

            output_path=
                output_path
        )
    )

    assert report is not None

    assert os.path.exists(
        output_path
    )

    assert (
        report[
            "model"
        ][
            "algorithm"
        ]
        ==
        "KMeans"
    )

    assert (
        report[
            "model"
        ][
            "selected_clusters"
        ]
        ==
        2
    )

    assert (
        report[
            "dataset"
        ][
            "total_customers"
        ]
        ==
        4
    )

    assert (
        report[
            "dataset"
        ][
            "feature_count"
        ]
        ==
        6
    )

    assert (
        report[
            "evaluation"
        ][
            "silhouette_score"
        ]
        ==
        0.52
    )

    assert (
        len(
            report[
                "segments"
            ]
        )
        ==
        2
    )

    with open(
        output_path,
        "r",
        encoding="utf-8"
    ) as file:

        saved_report = (
            json.load(
                file
            )
        )

    assert (
        saved_report[
            "model"
        ][
            "selected_clusters"
        ]
        ==
        2
    )

    assert (
        saved_report[
            "segments"
        ][0][
            "segment_name"
        ]
        ==
        "High Value Loyal"
    )