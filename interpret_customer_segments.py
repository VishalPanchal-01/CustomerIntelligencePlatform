import os

import pandas as pd

from src.analysis.segment_interpretation import (
    SegmentInterpreter
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

    summary_path = os.path.join(
        output_directory,
        "segment_summary.csv"
    )

    customers_path = os.path.join(
        output_directory,
        "customer_segments.csv"
    )

    # ---------------------------------
    # Load clustered customers
    # ---------------------------------

    clustered_df = pd.read_csv(
        input_path
    )

    # ---------------------------------
    # Create segment interpretation
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
    # Cluster → Segment mapping
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

    # ---------------------------------
    # Add segment names
    # to individual customers
    # ---------------------------------

    customer_segments = (
        clustered_df.copy()
    )

    customer_segments[
        "SegmentName"
    ] = (
        customer_segments[
            "Cluster"
        ]
        .map(
            segment_mapping
        )
    )

    # ---------------------------------
    # Add recommendations
    # ---------------------------------

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

    customer_segments[
        "Recommendation"
    ] = (
        customer_segments[
            "Cluster"
        ]
        .map(
            recommendation_mapping
        )
    )

    # ---------------------------------
    # Create artifact directory
    # ---------------------------------

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # ---------------------------------
    # Save outputs
    # ---------------------------------

    segment_summary.to_csv(
        summary_path,
        index=False
    )

    customer_segments.to_csv(
        customers_path,
        index=False
    )

    # ---------------------------------
    # Display results
    # ---------------------------------

    print(
        "\n================================"
    )

    print(
        "CUSTOMER SEGMENT INTERPRETATION"
    )

    print(
        "================================"
    )

    print(
        "\nSegment Summary:"
    )

    display_columns = [
        "Cluster",
        "SegmentName",
        "CustomerCount",
        "CustomerPercentage",
        "Recency",
        "Frequency",
        "Monetary",
        "AverageOrderValue",
        "Tenure"
    ]

    print(
        segment_summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nBusiness Recommendations:"
    )

    for _, row in (
        segment_summary.iterrows()
    ):

        print(
            f"\n{row['SegmentName']}:"
        )

        print(
            row[
                "Recommendation"
            ]
        )

    print(
        f"\nSegment summary saved to:"
        f"\n{summary_path}"
    )

    print(
        f"\nCustomer segments saved to:"
        f"\n{customers_path}"
    )


if __name__ == "__main__":

    main()