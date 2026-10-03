import json
import os

import pandas as pd

from src.evaluation.final_recommendation_report import (
    FinalRecommendationReport
)


def main():

    # =================================
    # PATHS
    # =================================

    selection_path = (
        "artifacts/recommendation/final/"
        "recommendation_final_selection.json"
    )

    production_metadata_path = (
        "models/recommendation/"
        "recommender_metadata.json"
    )

    comparison_path = (
        "artifacts/recommendation/"
        "recommendation_model_comparison.csv"
    )

    split_summary_path = (
        "artifacts/recommendation/"
        "recommendation_split_summary.json"
    )

    output_path = (
        "artifacts/recommendation/"
        "recommendation_model_report.json"
    )

    # =================================
    # CHECK FILES
    # =================================

    required_files = [
        selection_path,
        production_metadata_path,
        comparison_path,
        split_summary_path
    ]

    for file_path in required_files:

        if not os.path.exists(
            file_path
        ):

            raise FileNotFoundError(
                f"Required recommendation artifact "
                f"not found: {file_path}"
            )

    # =================================
    # LOAD JSON FILES
    # =================================

    with open(
        selection_path,
        "r",
        encoding="utf-8"
    ) as file:

        selection_metadata = (
            json.load(
                file
            )
        )

    with open(
        production_metadata_path,
        "r",
        encoding="utf-8"
    ) as file:

        production_metadata = (
            json.load(
                file
            )
        )

    with open(
        split_summary_path,
        "r",
        encoding="utf-8"
    ) as file:

        split_summary = (
            json.load(
                file
            )
        )

    # =================================
    # LOAD MODEL COMPARISON
    # =================================

    model_comparison = pd.read_csv(
        comparison_path
    )

    # =================================
    # GENERATE REPORT
    # =================================

    report_generator = (
        FinalRecommendationReport()
    )

    report = (
        report_generator.generate_report(
            selection_metadata=
                selection_metadata,

            production_metadata=
                production_metadata,

            model_comparison=
                model_comparison,

            split_summary=
                split_summary,

            output_path=
                output_path
        )
    )

    # =================================
    # DISPLAY FINAL SUMMARY
    # =================================

    selected = (
        report[
            "selected_strategy"
        ]
    )

    print(
        "\n================================"
    )

    print(
        "FINAL RECOMMENDATION REPORT"
    )

    print(
        "================================"
    )

    print(
        f"\nSelected Model: "
        f"{selected['model']}"
    )

    print(
        f"Selection K: "
        f"{selected['selection_k']}"
    )

    print(
        "\n--------------------------------"
    )

    print(
        "FINAL EVALUATION METRICS"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Precision@{selected['selection_k']}: "
        f"{selected['precision_at_k']:.4f}"
    )

    print(
        f"Recall@{selected['selection_k']}: "
        f"{selected['recall_at_k']:.4f}"
    )

    print(
        f"HitRate@{selected['selection_k']}: "
        f"{selected['hit_rate_at_k']:.4f}"
    )

    print(
        f"Catalog Coverage: "
        f"{selected['catalog_coverage']:.4f}"
    )

    print(
        f"Unique Recommended Products: "
        f"{selected['unique_recommended_products']}"
    )

    print(
        "\n--------------------------------"
    )

    print(
        "PRODUCTION MODEL"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Model Path: "
        f"{report['production_prediction']['model_path']}"
    )

    print(
        f"Default Top-K: "
        f"{report['production_prediction']['default_top_k']}"
    )

    print(
        "\nModes:"
    )

    print(
        "- next_purchase"
    )

    print(
        "- discovery"
    )

    print(
        f"\nFinal report saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":

    main()