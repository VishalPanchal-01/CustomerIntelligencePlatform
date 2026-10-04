import json
import os

import joblib
import pandas as pd

from src.dashboard.customer_intelligence_generation import (
    CustomerIntelligenceGenerator
)

from src.dashboard.unified_customer_dataset import (
    UnifiedCustomerDatasetBuilder
)

from src.prediction.clv_predictor import (
    CLVPredictor
)

from src.prediction.recommendation_predictor import (
    RecommendationPredictor
)


def main():

    # =========================================================
    # SOURCE CUSTOMER DATA
    # =========================================================

    customer_path = (
        "artifacts/segmentation/"
        "customer_segments.csv"
    )

    # =========================================================
    # MODEL PATHS
    # =========================================================

    churn_model_path = (
        "models/churn/"
        "churn_model.pkl"
    )

    clv_model_path = (
        "models/clv/"
        "clv_model.pkl"
    )

    clv_band_path = (
        "models/clv/"
        "clv_value_bands.json"
    )

    recommendation_model_path = (
        "models/recommendation/"
        "recommender.pkl"
    )

    recommendation_metadata_path = (
        "models/recommendation/"
        "recommender_metadata.json"
    )

    # =========================================================
    # MODULE OUTPUT PATHS
    # =========================================================

    churn_output_directory = (
        "artifacts/churn/"
        "predictions"
    )

    churn_output_path = os.path.join(
        churn_output_directory,
        "customer_churn_predictions.csv"
    )

    clv_output_directory = (
        "artifacts/clv/"
        "predictions"
    )

    clv_output_path = os.path.join(
        clv_output_directory,
        "customer_clv_predictions.csv"
    )

    recommendation_output_directory = (
        "artifacts/recommendation/"
        "predictions"
    )

    recommendation_output_path = os.path.join(
        recommendation_output_directory,
        "batch_recommendations.csv"
    )

    # =========================================================
    # DASHBOARD OUTPUTS
    # =========================================================

    dashboard_output_directory = (
        "artifacts/dashboard"
    )

    unified_output_path = os.path.join(
        dashboard_output_directory,
        "unified_customer_intelligence.csv"
    )

    coverage_output_path = os.path.join(
        dashboard_output_directory,
        "customer_intelligence_coverage.csv"
    )

    quality_output_path = os.path.join(
        dashboard_output_directory,
        "customer_intelligence_quality.json"
    )

    generation_summary_path = os.path.join(
        dashboard_output_directory,
        "customer_intelligence_generation_summary.json"
    )

    # =========================================================
    # VALIDATE REQUIRED FILES
    # =========================================================

    required_files = [
        customer_path,
        churn_model_path,
        clv_model_path,
        clv_band_path,
        recommendation_model_path,
        recommendation_metadata_path
    ]

    for file_path in required_files:

        if not os.path.exists(
            file_path
        ):

            raise FileNotFoundError(
                f"Required file not found:"
                f"\n{file_path}"
            )

    # =========================================================
    # CREATE OUTPUT DIRECTORIES
    # =========================================================

    directories = [
        churn_output_directory,
        clv_output_directory,
        recommendation_output_directory,
        dashboard_output_directory
    ]

    for directory in directories:

        os.makedirs(
            directory,
            exist_ok=True
        )

    # =========================================================
    # LOAD CUSTOMER POPULATION
    # =========================================================

    print(
        "\n========================================"
    )

    print(
        "GENERATING ALL CUSTOMER INTELLIGENCE"
    )

    print(
        "========================================"
    )

    customers = pd.read_csv(
        customer_path
    )

    generator = (
        CustomerIntelligenceGenerator(
            churn_low_threshold=0.30,
            churn_high_threshold=0.70
        )
    )

    customers = (
        generator.normalize_customer_id(
            customers
        )
    )

    total_customers = (
        customers[
            "Customer ID"
        ]
        .nunique()
    )

    print(
        f"\nCustomer population: "
        f"{total_customers}"
    )

    # =========================================================
    # LOAD CHURN MODEL
    # =========================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "1. CHURN PREDICTION"
    )

    print(
        "----------------------------------------"
    )

    churn_model = (
        joblib.load(
            churn_model_path
        )
    )

    churn_predictions = (
        generator.generate_churn_predictions(
            customer_df=
                customers,

            churn_model=
                churn_model
        )
    )

    churn_predictions.to_csv(
        churn_output_path,
        index=False
    )

    print(
        f"\nCustomers predicted: "
        f"{churn_predictions['Customer ID'].nunique()}"
    )

    print(
        f"Saved:"
        f"\n{churn_output_path}"
    )

    # =========================================================
    # CLV
    # =========================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "2. CLV PREDICTION"
    )

    print(
        "----------------------------------------"
    )

    clv_predictor = (
        CLVPredictor(
            model_path=
                clv_model_path,

            value_band_path=
                clv_band_path
        )
    )

    clv_predictions = (
        generator.generate_clv_predictions(
            customer_df=
                customers,

            clv_predictor=
                clv_predictor
        )
    )

    clv_predictions.to_csv(
        clv_output_path,
        index=False
    )

    print(
        f"\nCustomers predicted: "
        f"{clv_predictions['Customer ID'].nunique()}"
    )

    print(
        f"Saved:"
        f"\n{clv_output_path}"
    )

    # =========================================================
    # RECOMMENDATION
    # =========================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "3. PRODUCT RECOMMENDATIONS"
    )

    print(
        "----------------------------------------"
    )

    recommendation_predictor = (
        RecommendationPredictor(
            model_path=
                recommendation_model_path,

            metadata_path=
                recommendation_metadata_path
        )
    )

    recommendations = (
        generator.generate_recommendations(
            customer_df=
                customers,

            recommendation_predictor=
                recommendation_predictor,

            top_k=
                5,

            mode=
                "next_purchase"
        )
    )

    recommendations.to_csv(
        recommendation_output_path,
        index=False
    )

    print(
        f"\nCustomers receiving recommendations: "
        f"{recommendations['Customer ID'].nunique()}"
    )

    print(
        f"Recommendation rows: "
        f"{len(recommendations)}"
    )

    print(
        f"Saved:"
        f"\n{recommendation_output_path}"
    )

    # =========================================================
    # GENERATION SUMMARY
    # =========================================================

    generation_summary = (
        generator.create_generation_summary(
            customer_df=
                customers,

            churn_df=
                churn_predictions,

            clv_df=
                clv_predictions,

            recommendation_df=
                recommendations
        )
    )

    with open(
        generation_summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            generation_summary,
            file,
            indent=4
        )

    # =========================================================
    # BUILD UNIFIED DATASET
    # =========================================================

    print(
        "\n----------------------------------------"
    )

    print(
        "4. UNIFIED CUSTOMER DATASET"
    )

    print(
        "----------------------------------------"
    )

    unified_builder = (
        UnifiedCustomerDatasetBuilder()
    )

    unified = (
        unified_builder.build(
            segmentation_df=
                customers,

            churn_df=
                churn_predictions,

            clv_df=
                clv_predictions,

            recommendation_df=
                recommendations
        )
    )

    # =========================================================
    # COVERAGE
    # =========================================================

    coverage = (
        unified_builder.analyze_coverage(
            unified
        )
    )

    # =========================================================
    # QUALITY
    # =========================================================

    quality = (
        unified_builder.analyze_quality(
            unified
        )
    )

    # =========================================================
    # SAVE DASHBOARD ARTIFACTS
    # =========================================================

    unified.to_csv(
        unified_output_path,
        index=False
    )

    coverage.to_csv(
        coverage_output_path,
        index=False
    )

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
    # FINAL SUMMARY
    # =========================================================

    print(
        "\n========================================"
    )

    print(
        "GENERATION SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"\nTotal Customers: "
        f"{generation_summary['total_customers']}"
    )

    print(
        "\nChurn Coverage:"
    )

    print(
        f"{generation_summary['churn']['coverage_percentage']:.2f}%"
    )

    print(
        "\nCLV Coverage:"
    )

    print(
        f"{generation_summary['clv']['coverage_percentage']:.2f}%"
    )

    print(
        "\nRecommendation Coverage:"
    )

    print(
        f"{generation_summary['recommendation']['coverage_percentage']:.2f}%"
    )

    # =========================================================
    # MODULE COVERAGE
    # =========================================================

    print(
        "\n========================================"
    )

    print(
        "UNIFIED MODULE COVERAGE"
    )

    print(
        "========================================"
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
        "\n========================================"
    )

    print(
        "UNIFIED DATA QUALITY"
    )

    print(
        "========================================"
    )

    print(
        f"\nRows: "
        f"{quality['row_count']}"
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
    # FILES
    # =========================================================

    print(
        "\n========================================"
    )

    print(
        "FINAL OUTPUT FILES"
    )

    print(
        "========================================"
    )

    print(
        f"\nChurn:"
        f"\n{churn_output_path}"
    )

    print(
        f"\nCLV:"
        f"\n{clv_output_path}"
    )

    print(
        f"\nRecommendations:"
        f"\n{recommendation_output_path}"
    )

    print(
        f"\nUnified Customer Intelligence:"
        f"\n{unified_output_path}"
    )

    print(
        f"\nCoverage:"
        f"\n{coverage_output_path}"
    )

    print(
        f"\nQuality:"
        f"\n{quality_output_path}"
    )

    print(
        f"\nGeneration Summary:"
        f"\n{generation_summary_path}"
    )


if __name__ == "__main__":

    main()