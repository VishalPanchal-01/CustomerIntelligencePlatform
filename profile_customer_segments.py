import os

import pandas as pd

from src.analysis.segmentation_profile import (
    SegmentationProfiler
)


def main():

    # ---------------------------------
    # Paths
    # ---------------------------------

    input_path = (
        "artifacts/segmentation/"
        "clustered_customers.csv"
    )

    output_directory = (
        "artifacts/segmentation"
    )

    output_path = os.path.join(
        output_directory,
        "cluster_profile.csv"
    )

    # ---------------------------------
    # Load clustered customers
    # ---------------------------------

    df = pd.read_csv(
        input_path
    )

    # ---------------------------------
    # Create cluster profile
    # ---------------------------------

    profiler = (
        SegmentationProfiler()
    )

    profile = (
        profiler
        .create_cluster_profile(
            df
        )
    )

    # ---------------------------------
    # Save profile
    # ---------------------------------

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    profile.to_csv(
        output_path,
        index=False
    )

    # ---------------------------------
    # Console output
    # ---------------------------------

    print(
        "\n================================"
    )

    print(
        "CUSTOMER SEGMENT PROFILE"
    )

    print(
        "================================"
    )

    print(
        "\nCluster Profile:"
    )

    print(
        profile.to_string(
            index=False
        )
    )

    print(
        f"\nProfile saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":

    main()