import json
import os

import pandas as pd

from src.analysis.clv_analysis import (
    CLVAnalysis
)


def main():

    # ---------------------------------
    # Paths
    # ---------------------------------

    input_path = (
        "artifacts/clv/"
        "customer_clv_dataset.csv"
    )

    output_directory = (
        "artifacts/clv/analysis"
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # ---------------------------------
    # Load CLV dataset
    # ---------------------------------

    df = pd.read_csv(
        input_path
    )

    # ---------------------------------
    # Initialize analysis
    # ---------------------------------

    analyzer = (
        CLVAnalysis()
    )

    # ---------------------------------
    # Target statistics
    # ---------------------------------

    target_statistics = (
        analyzer
        .analyze_target_statistics(
            df
        )
    )

    # ---------------------------------
    # Zero revenue analysis
    # ---------------------------------

    zero_revenue = (
        analyzer
        .analyze_zero_revenue(
            df
        )
    )

    # ---------------------------------
    # Quantile analysis
    # ---------------------------------

    target_quantiles = (
        analyzer
        .analyze_target_quantiles(
            df
        )
    )

    # ---------------------------------
    # Outlier analysis
    # ---------------------------------

    target_outliers = (
        analyzer
        .analyze_target_outliers(
            df
        )
    )

    # ---------------------------------
    # Feature statistics
    # ---------------------------------

    feature_statistics = (
        analyzer
        .analyze_feature_statistics(
            df
        )
    )

    # ---------------------------------
    # Feature correlations
    # ---------------------------------

    feature_correlation = (
        analyzer
        .analyze_feature_correlation(
            df
        )
    )

    # ---------------------------------
    # Data quality
    # ---------------------------------

    data_quality = (
        analyzer
        .analyze_data_quality(
            df
        )
    )

    # ---------------------------------
    # Save analysis files
    # ---------------------------------

    target_statistics.to_csv(
        os.path.join(
            output_directory,
            "target_statistics.csv"
        ),
        index=False
    )

    zero_revenue.to_csv(
        os.path.join(
            output_directory,
            "zero_revenue_analysis.csv"
        ),
        index=False
    )

    target_quantiles.to_csv(
        os.path.join(
            output_directory,
            "target_quantiles.csv"
        ),
        index=False
    )

    target_outliers.to_csv(
        os.path.join(
            output_directory,
            "target_outliers.csv"
        ),
        index=False
    )

    feature_statistics.to_csv(
        os.path.join(
            output_directory,
            "feature_statistics.csv"
        ),
        index=False
    )

    feature_correlation.to_csv(
        os.path.join(
            output_directory,
            "feature_correlation.csv"
        ),
        index=False
    )

    with open(
        os.path.join(
            output_directory,
            "data_quality.json"
        ),
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data_quality,
            file,
            indent=4
        )

    # ---------------------------------
    # Console output
    # ---------------------------------

    print(
        "\n================================"
    )

    print(
        "CLV TARGET ANALYSIS"
    )

    print(
        "================================"
    )

    print(
        "\n1. Future Revenue Statistics:"
    )

    print(
        target_statistics.to_string(
            index=False
        )
    )

    print(
        "\n2. Zero Revenue Analysis:"
    )

    print(
        zero_revenue.to_string(
            index=False
        )
    )

    print(
        "\n3. Future Revenue Quantiles:"
    )

    print(
        target_quantiles.to_string(
            index=False
        )
    )

    print(
        "\n4. Future Revenue Outliers:"
    )

    print(
        target_outliers.to_string(
            index=False
        )
    )

    print(
        "\n5. Feature Statistics:"
    )

    print(
        feature_statistics.to_string(
            index=False
        )
    )

    print(
        "\n6. Feature Correlation "
        "with Future Revenue:"
    )

    print(
        feature_correlation.to_string(
            index=False
        )
    )

    print(
        "\n7. Data Quality:"
    )

    print(
        json.dumps(
            data_quality,
            indent=4
        )
    )

    print(
        f"\nAnalysis files saved to:"
        f"\n{output_directory}"
    )


if __name__ == "__main__":

    main()