import os

import pandas as pd

from src.preprocessing.segmentation_preprocessing import (
    SegmentationPreprocessor
)

from src.evaluation.segmentation_cluster_analysis import (
    SegmentationClusterAnalysis
)


def main():

    input_path = (
        "artifacts/segmentation/"
        "customer_segmentation_dataset.csv"
    )

    output_directory = (
        "artifacts/segmentation"
    )

    output_path = os.path.join(
        output_directory,
        "cluster_analysis.csv"
    )

    # ---------------------------------
    # Load segmentation dataset
    # ---------------------------------

    df = pd.read_csv(
        input_path
    )

    # ---------------------------------
    # Preprocess features
    # ---------------------------------

    preprocessor = (
        SegmentationPreprocessor()
    )

    (
        customer_ids,
        scaled_features
    ) = preprocessor.prepare_features(
        df
    )

    # ---------------------------------
    # Analyze possible cluster counts
    # ---------------------------------

    analyzer = (
        SegmentationClusterAnalysis()
    )

    report = (
        analyzer.evaluate_clusters(
            scaled_features,
            min_clusters=2,
            max_clusters=8
        )
    )

    # ---------------------------------
    # Save report
    # ---------------------------------

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    report.to_csv(
        output_path,
        index=False
    )

    # ---------------------------------
    # Find highest silhouette
    # ---------------------------------

    best_row = (
        report.loc[
            report[
                "SilhouetteScore"
            ].idxmax()
        ]
    )

    best_k = int(
        best_row["K"]
    )

    best_score = float(
        best_row[
            "SilhouetteScore"
        ]
    )

    # ---------------------------------
    # Console output
    # ---------------------------------

    print(
        "\n================================="
    )

    print(
        "SEGMENTATION CLUSTER ANALYSIS"
    )

    print(
        "================================="
    )

    print(
        "\nCluster Analysis:"
    )

    print(
        report.to_string(
            index=False
        )
    )

    print(
        "\nBest K by Silhouette Score:"
    )

    print(
        best_k
    )

    print(
        "\nBest Silhouette Score:"
    )

    print(
        f"{best_score:.4f}"
    )

    print(
        f"\nAnalysis saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":

    main()