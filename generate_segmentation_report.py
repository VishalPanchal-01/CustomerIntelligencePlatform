import os

import pandas as pd

from sklearn.metrics import silhouette_score

from src.preprocessing.segmentation_preprocessing import (
    SegmentationPreprocessor
)

from src.training.kmeans_model import (
    CustomerSegmentationModel
)

from src.analysis.segment_interpretation import (
    SegmentInterpreter
)

from src.visualization.segmentation_visualization import (
    SegmentationVisualization
)

from src.utils.segmentation_persistence import (
    SegmentationPersistence
)

from src.evaluation.final_segmentation_report import (
    FinalSegmentationReport
)


def main():

    # ---------------------------------
    # Configuration
    # ---------------------------------

    input_path = (
        "artifacts/segmentation/"
        "customer_segmentation_dataset.csv"
    )

    output_directory = (
        "artifacts/segmentation"
    )

    model_directory = (
        "models/segmentation"
    )

    model_path = os.path.join(
        model_directory,
        "kmeans_model.pkl"
    )

    preprocessor_path = os.path.join(
        model_directory,
        "segmentation_preprocessor.pkl"
    )

    segmented_path = os.path.join(
        output_directory,
        "customer_segments.csv"
    )

    summary_path = os.path.join(
        output_directory,
        "segment_summary.csv"
    )

    report_path = os.path.join(
        output_directory,
        "segmentation_model_report.json"
    )

    # IMPORTANT:
    # Use the K you selected from
    # cluster analysis.
    selected_k = 4

    # ---------------------------------
    # Features
    # ---------------------------------

    feature_names = [
        "Recency",
        "Frequency",
        "Monetary",
        "TotalItems",
        "AverageOrderValue",
        "Tenure"
    ]

    # ---------------------------------
    # Load dataset
    # ---------------------------------

    df = pd.read_csv(
        input_path
    )

    # ---------------------------------
    # Preprocess
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
    # Train final K-Means
    # ---------------------------------

    trainer = (
        CustomerSegmentationModel(
            n_clusters=selected_k,
            random_state=42
        )
    )

    model = trainer.train(
        scaled_features
    )

    # ---------------------------------
    # Assign clusters
    # ---------------------------------

    clusters = (
        trainer.assign_clusters(
            model,
            scaled_features
        )
    )

    # ---------------------------------
    # Final evaluation
    # ---------------------------------

    final_silhouette_score = (
        silhouette_score(
            scaled_features,
            clusters
        )
    )

    final_inertia = float(
        model.inertia_
    )

    # ---------------------------------
    # Build clustered dataset
    # ---------------------------------

    clustered_df = (
        df.copy()
    )

    clustered_df[
        "Cluster"
    ] = clusters.to_numpy()

    # ---------------------------------
    # Segment interpretation
    # ---------------------------------

    interpreter = (
        SegmentInterpreter()
    )

    segment_summary = (
        interpreter
        .create_segment_summary(
            clustered_df
        )
    )

    # ---------------------------------
    # Create mapping
    # ---------------------------------

    segment_mapping = (
        segment_summary[
            [
                "Cluster",
                "SegmentName"
            ]
        ]
        .set_index(
            "Cluster"
        )[
            "SegmentName"
        ]
        .to_dict()
    )

    recommendation_mapping = (
        segment_summary[
            [
                "Cluster",
                "Recommendation"
            ]
        ]
        .set_index(
            "Cluster"
        )[
            "Recommendation"
        ]
        .to_dict()
    )

    # ---------------------------------
    # Add business information
    # ---------------------------------

    clustered_df[
        "SegmentName"
    ] = (
        clustered_df[
            "Cluster"
        ]
        .map(
            segment_mapping
        )
    )

    clustered_df[
        "Recommendation"
    ] = (
        clustered_df[
            "Cluster"
        ]
        .map(
            recommendation_mapping
        )
    )

    # ---------------------------------
    # Save datasets
    # ---------------------------------

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    clustered_df.to_csv(
        segmented_path,
        index=False
    )

    segment_summary.to_csv(
        summary_path,
        index=False
    )

    # ---------------------------------
    # Save model artifacts
    # ---------------------------------

    persistence = (
        SegmentationPersistence()
    )

    persistence.save_artifact(
        preprocessor,
        preprocessor_path
    )

    persistence.save_artifact(
        model,
        model_path
    )

    # ---------------------------------
    # Create visualizations
    # ---------------------------------

    visualizer = (
        SegmentationVisualization()
    )

    distribution_path = (
        visualizer
        .plot_cluster_distribution(
            clustered_df
        )
    )

    feature_profile_path = (
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

    visualization_paths = {

        "cluster_distribution":
            distribution_path,

        "cluster_feature_profile":
            feature_profile_path,

        "pca_visualization":
            pca_path
    }

    # ---------------------------------
    # Generate final report
    # ---------------------------------

    report_generator = (
        FinalSegmentationReport()
    )

    report = (
        report_generator
        .generate_report(
            segmented_df=
                clustered_df,

            segment_summary=
                segment_summary,

            silhouette_score_value=
                final_silhouette_score,

            inertia=
                final_inertia,

            selected_k=
                selected_k,

            feature_names=
                feature_names,

            preprocessor_path=
                preprocessor_path,

            model_path=
                model_path,

            visualization_paths=
                visualization_paths,

            output_path=
                report_path
        )
    )

    # ---------------------------------
    # Console output
    # ---------------------------------

    print(
        "\n================================="
    )

    print(
        "FINAL SEGMENTATION REPORT"
    )

    print(
        "================================="
    )

    print(
        f"\nAlgorithm: "
        f"{report['model']['algorithm']}"
    )

    print(
        f"\nSelected K: "
        f"{report['model']['selected_clusters']}"
    )

    print(
        f"\nTotal Customers: "
        f"{report['dataset']['total_customers']}"
    )

    print(
        f"\nSilhouette Score: "
        f"{report['evaluation']['silhouette_score']:.4f}"
    )

    print(
        f"\nInertia: "
        f"{report['evaluation']['inertia']:.4f}"
    )

    print(
        "\nSegments:"
    )

    for segment in (
        report["segments"]
    ):

        print(
            f"\nCluster "
            f"{segment['cluster']}"
        )

        print(
            f"Segment: "
            f"{segment['segment_name']}"
        )

        print(
            f"Customers: "
            f"{segment['customer_count']}"
        )

        print(
            f"Percentage: "
            f"{segment['customer_percentage']:.2f}%"
        )

    print(
        f"\nModel saved to:"
        f"\n{model_path}"
    )

    print(
        f"\nPreprocessor saved to:"
        f"\n{preprocessor_path}"
    )

    print(
        f"\nSegmented customers saved to:"
        f"\n{segmented_path}"
    )

    print(
        f"\nFinal report saved to:"
        f"\n{report_path}"
    )


if __name__ == "__main__":

    main()