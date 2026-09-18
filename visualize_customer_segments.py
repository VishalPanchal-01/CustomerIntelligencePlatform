import pandas as pd

from src.preprocessing.segmentation_preprocessing import (
    SegmentationPreprocessor
)

from src.training.kmeans_model import (
    CustomerSegmentationModel
)

from src.visualization.segmentation_visualization import (
    SegmentationVisualization
)


def main():

    # ---------------------------------
    # Load segmentation dataset
    # ---------------------------------

    input_path = (
        "artifacts/segmentation/"
        "customer_segmentation_dataset.csv"
    )

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
    # Final cluster count
    # ---------------------------------

    final_k = 4

    # ---------------------------------
    # Train K-Means
    # ---------------------------------

    trainer = (
        CustomerSegmentationModel(
            n_clusters=final_k,
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
    # Create customer-level dataset
    # ---------------------------------

    clustered_df = df.copy()

    clustered_df[
        "Cluster"
    ] = clusters.to_numpy()

    # ---------------------------------
    # Initialize visualization
    # ---------------------------------

    visualizer = (
        SegmentationVisualization()
    )

    # ---------------------------------
    # Cluster distribution
    # ---------------------------------

    distribution_path = (
        visualizer
        .plot_cluster_distribution(
            clustered_df
        )
    )

    # ---------------------------------
    # Cluster feature profile
    # ---------------------------------

    profile_path = (
        visualizer
        .plot_cluster_feature_profile(
            clustered_df
        )
    )

    # ---------------------------------
    # PCA visualization
    # ---------------------------------

    pca_path = (
        visualizer
        .plot_pca_clusters(
            scaled_features,
            clusters
        )
    )

    # ---------------------------------
    # Display output paths
    # ---------------------------------

    print(
        "\n================================"
    )

    print(
        "SEGMENTATION VISUALIZATIONS"
    )

    print(
        "================================"
    )

    print(
        f"\nCluster Distribution:"
        f"\n{distribution_path}"
    )

    print(
        f"\nFeature Profile:"
        f"\n{profile_path}"
    )

    print(
        f"\nPCA Visualization:"
        f"\n{pca_path}"
    )


if __name__ == "__main__":

    main()