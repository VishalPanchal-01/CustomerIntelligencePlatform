import os

import pandas as pd

from sklearn.metrics import silhouette_score

from src.preprocessing.segmentation_preprocessing import (
    SegmentationPreprocessor
)

from src.training.kmeans_model import (
    CustomerSegmentationModel
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

    output_path = os.path.join(
        output_directory,
        "clustered_customers.csv"
    )

    final_k = 4

    # ---------------------------------
    # Load dataset
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
    # Train final K-Means model
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
    # Evaluate final clustering
    # ---------------------------------

    silhouette = (
        silhouette_score(
            scaled_features,
            clusters
        )
    )

    # ---------------------------------
    # Add cluster labels
    # to original customer dataset
    # ---------------------------------

    clustered_df = (
        df.copy()
    )

    clustered_df[
        "Cluster"
    ] = clusters.to_numpy()

    # ---------------------------------
    # Create output directory
    # ---------------------------------

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # ---------------------------------
    # Save clustered customers
    # ---------------------------------

    clustered_df.to_csv(
        output_path,
        index=False
    )

    # ---------------------------------
    # Cluster counts
    # ---------------------------------

    cluster_counts = (
        clustered_df[
            "Cluster"
        ]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------
    # Console output
    # ---------------------------------

    print(
        "\n================================"
    )

    print(
        "FINAL CUSTOMER SEGMENTATION"
    )

    print(
        "================================"
    )

    print(
        f"\nNumber of Clusters: "
        f"{final_k}"
    )

    print(
        f"\nSilhouette Score: "
        f"{silhouette:.4f}"
    )

    print(
        "\nCluster Distribution:"
    )

    print(
        cluster_counts
    )

    print(
        "\nClustered Customer Dataset:"
    )

    print(
        clustered_df.head()
    )

    print(
        f"\nSaved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":

    main()