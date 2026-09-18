import os

import pandas as pd

from src.preprocessing.segmentation_preprocessing import (
    SegmentationPreprocessor
)

from src.training.kmeans_model import (
    CustomerSegmentationModel
)

from src.utils.segmentation_persistence import (
    SegmentationPersistence
)


def main():

    # ---------------------------------
    # Configuration
    # ---------------------------------

    input_path = (
        "artifacts/segmentation/"
        "customer_segmentation_dataset.csv"
    )

    model_directory = (
        "models/segmentation"
    )

    preprocessor_path = os.path.join(
        model_directory,
        "segmentation_preprocessor.pkl"
    )

    kmeans_path = os.path.join(
        model_directory,
        "kmeans_model.pkl"
    )

    # Replace this with your
    # selected cluster count
    final_k = 4

    # ---------------------------------
    # Load segmentation dataset
    # ---------------------------------

    df = pd.read_csv(
        input_path
    )

    # ---------------------------------
    # Fit preprocessing
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
    # Save artifacts
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
        kmeans_path
    )

    # ---------------------------------
    # Console output
    # ---------------------------------

    print(
        "\n================================"
    )

    print(
        "SEGMENTATION ARTIFACTS SAVED"
    )

    print(
        "================================"
    )

    print(
        f"\nPreprocessor:"
        f"\n{preprocessor_path}"
    )

    print(
        f"\nK-Means Model:"
        f"\n{kmeans_path}"
    )

    print(
        f"\nNumber of Clusters: "
        f"{final_k}"
    )


if __name__ == "__main__":

    main()