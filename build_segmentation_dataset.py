import os

import pandas as pd

from src.feature_engineering.segmentation_features import (
    SegmentationFeatureEngineering
)


def main():

    # ---------------------------------
    # Input/output paths
    # ---------------------------------

    input_path = os.path.join(
        "artifacts",
        "churn",
        "customer_churn_dataset.csv"
    )

    output_directory = os.path.join(
        "artifacts",
        "segmentation"
    )

    output_path = os.path.join(
        output_directory,
        "customer_segmentation_dataset.csv"
    )

    # ---------------------------------
    # Check source dataset exists
    # ---------------------------------

    if not os.path.exists(
        input_path
    ):

        raise FileNotFoundError(
            f"Input churn dataset not found: "
            f"{input_path}"
        )

    # ---------------------------------
    # Load customer dataset
    # ---------------------------------

    df = pd.read_csv(
        input_path
    )

    print(
        "\nSource Dataset Shape:"
    )

    print(
        df.shape
    )

    print(
        "\nSource Dataset Columns:"
    )

    print(
        df.columns.tolist()
    )

    # ---------------------------------
    # Create segmentation features
    # ---------------------------------

    feature_engineering = (
        SegmentationFeatureEngineering()
    )

    segmentation_df = (
        feature_engineering
        .create_segmentation_features(
            df
        )
    )

    # ---------------------------------
    # Create output directory
    # ---------------------------------

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # ---------------------------------
    # Save dataset
    # ---------------------------------

    segmentation_df.to_csv(
        output_path,
        index=False
    )

    # ---------------------------------
    # Show results
    # ---------------------------------

    print(
        "\nCustomer Segmentation Dataset:"
    )

    print(
        segmentation_df.head()
    )

    print(
        "\nSegmentation Dataset Shape:"
    )

    print(
        segmentation_df.shape
    )

    print(
        "\nSegmentation Columns:"
    )

    print(
        segmentation_df.columns.tolist()
    )

    print(
        "\nMissing Values:"
    )

    print(
        segmentation_df
        .isnull()
        .sum()
    )

    print(
        "\nDuplicate Customers:"
    )

    print(
        segmentation_df[
            "CustomerID"
        ]
        .duplicated()
        .sum()
    )

    print(
        "\nFeature Statistics:"
    )

    print(
        segmentation_df[
            [
                "Recency",
                "Frequency",
                "Monetary",
                "TotalItems",
                "AverageOrderValue",
                "Tenure"
            ]
        ]
        .describe()
    )

    print(
        f"\nDataset saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":

    main()