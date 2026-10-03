import json
import os

import pandas as pd

from src.evaluation.final_clv_report import (
    FinalCLVReport
)


def main():

    # ---------------------------------
    # Paths
    # ---------------------------------

    metadata_path = (
        "artifacts/clv/final/"
        "clv_final_model_metadata.json"
    )

    feature_importance_path = (
        "artifacts/clv/final/"
        "clv_feature_importance.csv"
    )

    value_band_path = (
        "models/clv/"
        "clv_value_bands.json"
    )

    output_path = (
        "artifacts/clv/"
        "clv_model_report.json"
    )

    # ---------------------------------
    # Validate files
    # ---------------------------------

    required_files = [
        metadata_path,
        feature_importance_path,
        value_band_path
    ]

    for file_path in required_files:

        if not os.path.exists(
            file_path
        ):

            raise FileNotFoundError(
                f"Required CLV artifact "
                f"not found: {file_path}"
            )

    # ---------------------------------
    # Load metadata
    # ---------------------------------

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(
            file
        )

    # ---------------------------------
    # Load feature importance
    # ---------------------------------

    feature_importance = pd.read_csv(
        feature_importance_path
    )

    # ---------------------------------
    # Load value bands
    # ---------------------------------

    with open(
        value_band_path,
        "r",
        encoding="utf-8"
    ) as file:

        value_bands = json.load(
            file
        )

    # ---------------------------------
    # Generate report
    # ---------------------------------

    generator = (
        FinalCLVReport()
    )

    report = (
        generator.generate_report(
            metadata=
                metadata,

            feature_importance=
                feature_importance,

            value_bands=
                value_bands,

            output_path=
                output_path
        )
    )

    # ---------------------------------
    # Console summary
    # ---------------------------------

    print(
        "\n================================"
    )

    print(
        "FINAL CLV REPORT"
    )

    print(
        "================================"
    )

    print(
        f"\nSelected Model: "
        f"{report['selected_model']['algorithm']}"
    )

    print(
        f"Target Strategy: "
        f"{report['selected_model']['target_strategy']}"
    )

    print(
        f"Best CV MAE: "
        f"{report['selected_model']['best_cross_validated_mae']:.4f}"
    )

    print(
        "\n--------------------------------"
    )

    print(
        "FINAL TEST PERFORMANCE"
    )

    print(
        "--------------------------------"
    )

    print(
        f"MAE: "
        f"{report['final_test_performance']['mae']:.4f}"
    )

    print(
        f"RMSE: "
        f"{report['final_test_performance']['rmse']:.4f}"
    )

    print(
        f"R²: "
        f"{report['final_test_performance']['r2']:.4f}"
    )

    print(
        "\n--------------------------------"
    )

    print(
        "TOP FEATURES"
    )

    print(
        "--------------------------------"
    )

    for feature in (
        report[
            "feature_importance"
        ][:3]
    ):

        print(
            f"{feature['feature']}: "
            f"{feature['importance_percentage']:.2f}%"
        )

    print(
        "\n--------------------------------"
    )

    print(
        "CLV VALUE BANDS"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Low <= "
        f"{report['value_bands']['low_upper_bound']:.2f}"
    )

    print(
        f"Medium <= "
        f"{report['value_bands']['medium_upper_bound']:.2f}"
    )

    print(
        f"High > "
        f"{report['value_bands']['medium_upper_bound']:.2f}"
    )

    print(
        f"\nFinal report saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":

    main()