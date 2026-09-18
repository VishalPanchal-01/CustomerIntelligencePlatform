import pandas as pd

from src.preprocessing.segmentation_preprocessing import (
    SegmentationPreprocessor
)


def main():

    input_path = (
        "artifacts/segmentation/"
        "customer_segmentation_dataset.csv"
    )

    # ---------------------------------
    # Load segmentation dataset
    # ---------------------------------

    df = pd.read_csv(
        input_path
    )

    # ---------------------------------
    # Initialize preprocessor
    # ---------------------------------

    preprocessor = (
        SegmentationPreprocessor()
    )

    # ---------------------------------
    # Prepare features
    # ---------------------------------

    (
        customer_ids,
        scaled_features
    ) = preprocessor.prepare_features(
        df
    )

    # ---------------------------------
    # Console output
    # ---------------------------------

    print(
        "\nCustomer IDs:"
    )

    print(
        customer_ids.head()
    )

    print(
        "\nScaled Segmentation Features:"
    )

    print(
        scaled_features.head()
    )

    print(
        "\nScaled Feature Shape:"
    )

    print(
        scaled_features.shape
    )

    print(
        "\nScaled Feature Means:"
    )

    print(
        scaled_features.mean()
    )

    print(
        "\nScaled Feature Standard Deviations:"
    )

    print(
        scaled_features.std()
    )


if __name__ == "__main__":

    main()