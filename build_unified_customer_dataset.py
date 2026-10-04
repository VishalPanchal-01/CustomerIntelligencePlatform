import json
import os

import pandas as pd

from src.dashboard.unified_customer_dataset import (
    UnifiedCustomerDatasetBuilder
)


# =============================================================
# OPTIONAL CSV LOADER
# =============================================================

def load_optional_csv(
    file_path: str
):

    if not os.path.exists(
        file_path
    ):

        print(
            f"[WARNING] Optional file not found:"
            f"\n{file_path}"
        )

        return None

    print(
        f"[INFO] Loading:"
        f"\n{file_path}"
    )

    return pd.read_csv(
        file_path
    )


# =============================================================
# MAIN
# =============================================================

def main():

    # =========================================================
    # INPUT PATHS
    # =========================================================

    segmentation_path = (
        "artifacts/segmentation/"
        "customer_segments.csv"
    )

    churn_path = (
        "artifacts/churn/"
        "predictions/"
        "customer_churn_predictions.csv"
    )

    clv_path = (
        "artifacts/clv/"
        "predictions/"
        "customer_clv_predictions.csv"
    )

    recommendation_path = (
        "artifacts/recommendation/"
        "predictions/"
        "batch_recommendations.csv"
    )

    # =========================================================
    # OUTPUT PATHS
    # =========================================================

    output_directory = (
        "artifacts/dashboard"
    )

    unified_output_path = os.path.join(
        output_directory,
        "unified_customer_intelligence.csv"
    )

    coverage_output_path = os.path.join(
        output_directory,
        "customer_intelligence_coverage.csv"
    )

    quality_output_path = os.path.join(
        output_directory,
        "customer_intelligence_quality.json"
    )

    # =========================================================
    # VALIDATE SEGMENTATION FILE
    # =========================================================

    if not os.path.exists(
        segmentation_path
    ):

        raise FileNotFoundError(
            f"Required segmentation file "
            f"not found:"
            f"\n{segmentation_path}"
        )

    # =========================================================
    # LOAD SEGMENTATION
    # =========================================================

    print(
        "\n================================"
    )

    print(
        "LOADING CUSTOMER INTELLIGENCE"
    )

    print(
        "================================"
    )

    segmentation_df = pd.read_csv(
        segmentation_path
    )

    print(
        f"\nSegmentation:"
        f"\nRows = {len(segmentation_df)}"
    )

    # =========================================================
    # LOAD OPTIONAL MODULES
    # =========================================================

    churn_df = (
        load_optional_csv(
            churn_path
        )
    )

    clv_df = (
        load_optional_csv(
            clv_path
        )
    )

    recommendation_df = (
        load_optional_csv(
            recommendation_path
        )
    )

    # =========================================================
    # CREATE BUILDER
    # =========================================================

    builder = (
        UnifiedCustomerDatasetBuilder()
    )

    # =========================================================
    # BUILD UNIFIED DATASET
    # =========================================================

    unified = (
        builder.build(
            segmentation_df=
                segmentation_df,

            churn_df=
                churn_df,

            clv_df=
                clv_df,

            recommendation_df=
                recommendation_df
        )
    )

    # =========================================================
    # COVERAGE ANALYSIS
    # =========================================================

    coverage = (
        builder.analyze_coverage(
            unified
        )
    )

    # =========================================================
    # QUALITY ANALYSIS
    # =========================================================

    quality = (
        builder.analyze_quality(
            unified
        )
    )

    # =========================================================
    # CREATE DIRECTORY
    # =========================================================

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # =========================================================
    # SAVE UNIFIED DATASET
    # =========================================================

    unified.to_csv(
        unified_output_path,
        index=False
    )

    # =========================================================
    # SAVE COVERAGE REPORT
    # =========================================================

    coverage.to_csv(
        coverage_output_path,
        index=False
    )

    # =========================================================
    # SAVE QUALITY REPORT
    # =========================================================

    with open(
        quality_output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            quality,
            file,
            indent=4
        )

    # =========================================================
    # CONSOLE SUMMARY
    # =========================================================

    print(
        "\n================================"
    )

    print(
        "UNIFIED CUSTOMER INTELLIGENCE"
    )

    print(
        "================================"
    )

    print(
        f"\nRows:"
        f"\n{len(unified)}"
    )

    print(
        f"\nUnique Customers:"
        f"\n{unified['Customer ID'].nunique()}"
    )

    print(
        f"\nColumns:"
        f"\n{len(unified.columns)}"
    )

    # =========================================================
    # DISPLAY COLUMNS
    # =========================================================

    print(
        "\n--------------------------------"
    )

    print(
        "DATASET COLUMNS"
    )

    print(
        "--------------------------------"
    )

    for column in unified.columns:

        print(
            f"- {column}"
        )

    # =========================================================
    # COVERAGE
    # =========================================================

    print(
        "\n================================"
    )

    print(
        "MODULE COVERAGE"
    )

    print(
        "================================"
    )

    print(
        "\n"
        + coverage.to_string(
            index=False
        )
    )

    # =========================================================
    # QUALITY
    # =========================================================

    print(
        "\n================================"
    )

    print(
        "DATA QUALITY"
    )

    print(
        "================================"
    )

    print(
        f"\nRows: "
        f"{quality['row_count']}"
    )

    print(
        f"Columns: "
        f"{quality['column_count']}"
    )

    print(
        f"Unique Customers: "
        f"{quality['unique_customers']}"
    )

    print(
        f"Duplicate Customers: "
        f"{quality['duplicate_customers']}"
    )

    print(
        f"Missing Customer IDs: "
        f"{quality['missing_customer_ids']}"
    )

    print(
        f"One Row Per Customer: "
        f"{quality['one_row_per_customer']}"
    )

    # =========================================================
    # SAMPLE DATA
    # =========================================================

    print(
        "\n================================"
    )

    print(
        "SAMPLE CUSTOMER INTELLIGENCE"
    )

    print(
        "================================"
    )

    print(
        "\n"
        + unified
        .head(
            10
        )
        .to_string(
            index=False
        )
    )

    # =========================================================
    # OUTPUT FILES
    # =========================================================

    print(
        "\n================================"
    )

    print(
        "OUTPUT FILES"
    )

    print(
        "================================"
    )

    print(
        f"\nUnified Dataset:"
        f"\n{unified_output_path}"
    )

    print(
        f"\nCoverage Report:"
        f"\n{coverage_output_path}"
    )

    print(
        f"\nQuality Report:"
        f"\n{quality_output_path}"
    )


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    main()