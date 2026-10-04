import json
import os

import pandas as pd

from src.explainability.explainability_validation import (
    ExplainabilityValidation
)


# ============================================================
# PATHS
# ============================================================

EXPLAINABILITY_ROOT = os.path.join(
    "artifacts",
    "explainability"
)

CHURN_DIRECTORY = os.path.join(
    EXPLAINABILITY_ROOT,
    "churn"
)

CLV_DIRECTORY = os.path.join(
    EXPLAINABILITY_ROOT,
    "clv"
)

DASHBOARD_DATASET_PATH = os.path.join(
    "artifacts",
    "dashboard",
    "unified_customer_intelligence.csv"
)


# ------------------------------------------------------------
# Churn paths
# ------------------------------------------------------------

CHURN_IMPORTANCE_PATH = os.path.join(
    CHURN_DIRECTORY,
    "churn_global_shap_importance.csv"
)

CHURN_LOCAL_PATH = os.path.join(
    CHURN_DIRECTORY,
    "churn_dashboard_shap_values.csv"
)

CHURN_SUMMARY_PATH = os.path.join(
    CHURN_DIRECTORY,
    "churn_visualization_summary.json"
)

CHURN_GLOBAL_PLOT = os.path.join(
    CHURN_DIRECTORY,
    "visualizations",
    "churn_global_feature_importance.png"
)

CHURN_BEESWARM_PLOT = os.path.join(
    CHURN_DIRECTORY,
    "visualizations",
    "churn_shap_beeswarm.png"
)

CHURN_DRIVER_PLOT = os.path.join(
    CHURN_DIRECTORY,
    "visualizations",
    "example_customer_churn_drivers.png"
)

CHURN_WATERFALL_PLOT = os.path.join(
    CHURN_DIRECTORY,
    "visualizations",
    "example_customer_churn_waterfall.png"
)


# ------------------------------------------------------------
# CLV paths
# ------------------------------------------------------------

CLV_IMPORTANCE_PATH = os.path.join(
    CLV_DIRECTORY,
    "clv_global_shap_importance.csv"
)

CLV_LOCAL_PATH = os.path.join(
    CLV_DIRECTORY,
    "clv_dashboard_shap_values.csv"
)

CLV_SUMMARY_PATH = os.path.join(
    CLV_DIRECTORY,
    "clv_visualization_summary.json"
)

CLV_GLOBAL_PLOT = os.path.join(
    CLV_DIRECTORY,
    "visualizations",
    "clv_global_feature_importance.png"
)

CLV_BEESWARM_PLOT = os.path.join(
    CLV_DIRECTORY,
    "visualizations",
    "clv_shap_beeswarm.png"
)

CLV_DRIVER_PLOT = os.path.join(
    CLV_DIRECTORY,
    "visualizations",
    "example_customer_clv_drivers.png"
)

CLV_WATERFALL_PLOT = os.path.join(
    CLV_DIRECTORY,
    "visualizations",
    "example_customer_clv_waterfall.png"
)


# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------

REPORT_PATH = os.path.join(
    EXPLAINABILITY_ROOT,
    "explainability_report.json"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "========================================"
    )

    print(
        "FINAL EXPLAINABILITY VALIDATION"
    )

    print(
        "========================================"
        "\n"
    )

    os.makedirs(
        EXPLAINABILITY_ROOT,
        exist_ok=True
    )

    validator = (
        ExplainabilityValidation()
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    churn_importance = (
        validator.load_csv(
            CHURN_IMPORTANCE_PATH
        )
    )

    churn_local = (
        validator.load_csv(
            CHURN_LOCAL_PATH
        )
    )

    clv_importance = (
        validator.load_csv(
            CLV_IMPORTANCE_PATH
        )
    )

    clv_local = (
        validator.load_csv(
            CLV_LOCAL_PATH
        )
    )

    dashboard_df = (
        validator.load_csv(
            DASHBOARD_DATASET_PATH
        )
    )

    churn_summary = (
        validator.load_json(
            CHURN_SUMMARY_PATH
        )
    )

    clv_summary = (
        validator.load_json(
            CLV_SUMMARY_PATH
        )
    )

    # ========================================================
    # FILE VALIDATION
    # ========================================================

    artifact_paths = {

        "churn_global_importance_csv":
            CHURN_IMPORTANCE_PATH,

        "churn_local_shap_csv":
            CHURN_LOCAL_PATH,

        "churn_summary_json":
            CHURN_SUMMARY_PATH,

        "churn_global_importance_png":
            CHURN_GLOBAL_PLOT,

        "churn_beeswarm_png":
            CHURN_BEESWARM_PLOT,

        "churn_driver_png":
            CHURN_DRIVER_PLOT,

        "churn_waterfall_png":
            CHURN_WATERFALL_PLOT,

        "clv_global_importance_csv":
            CLV_IMPORTANCE_PATH,

        "clv_local_shap_csv":
            CLV_LOCAL_PATH,

        "clv_summary_json":
            CLV_SUMMARY_PATH,

        "clv_global_importance_png":
            CLV_GLOBAL_PLOT,

        "clv_beeswarm_png":
            CLV_BEESWARM_PLOT,

        "clv_driver_png":
            CLV_DRIVER_PLOT,

        "clv_waterfall_png":
            CLV_WATERFALL_PLOT
    }

    file_validation = {}

    for name, path in artifact_paths.items():

        file_validation[
            name
        ] = (
            validator.check_file(
                path
            )
        )

    # ========================================================
    # DATA VALIDATION
    # ========================================================

    churn_importance_validation = (
        validator
        .validate_importance_data(
            churn_importance
        )
    )

    churn_local_validation = (
        validator
        .validate_local_shap_data(
            churn_local
        )
    )

    clv_importance_validation = (
        validator
        .validate_importance_data(
            clv_importance
        )
    )

    clv_local_validation = (
        validator
        .validate_local_shap_data(
            clv_local
        )
    )

    # ========================================================
    # COVERAGE
    # ========================================================

    churn_coverage = (
        validator
        .explanation_coverage(
            churn_local,
            dashboard_df
        )
    )

    clv_coverage = (
        validator
        .explanation_coverage(
            clv_local,
            dashboard_df
        )
    )

    common_customers = (
        validator
        .common_explained_customers(
            churn_local,
            clv_local
        )
    )

    # ========================================================
    # TOP GLOBAL FEATURES
    # ========================================================

    churn_top_features = (
        validator.top_features(
            churn_importance,
            top_n=6
        )
    )

    clv_top_features = (
        validator.top_features(
            clv_importance,
            top_n=6
        )
    )

    # ========================================================
    # OVERALL ARTIFACT STATUS
    # ========================================================

    all_files_exist = all(
        item[
            "exists"
        ]
        for item in file_validation.values()
    )

    core_data_valid = all(
        [
            churn_importance_validation[
                "valid"
            ],

            churn_local_validation[
                "valid"
            ],

            clv_importance_validation[
                "valid"
            ],

            clv_local_validation[
                "valid"
            ]
        ]
    )

    overall_status = (
        "PASS"
        if
        all_files_exist
        and
        core_data_valid
        else
        "REVIEW_REQUIRED"
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    report = {

        "phase":
            "Phase 6 — Explainable AI",

        "status":
            overall_status,

        "method":
            "SHAP",

        "modules_explained": [
            "Customer Churn Prediction",
            "Predicted 90-Day Revenue"
        ],

        "artifact_validation":
            file_validation,

        "dataset_validation": {

            "churn_global_importance":
                churn_importance_validation,

            "churn_local_explanations":
                churn_local_validation,

            "clv_global_importance":
                clv_importance_validation,

            "clv_local_explanations":
                clv_local_validation
        },

        "explanation_coverage": {

            "churn":
                churn_coverage,

            "clv":
                clv_coverage,

            "customers_explained_by_both":
                {
                    "count":
                        common_customers[
                            "count"
                        ]
                }
        },

        "global_feature_importance": {

            "churn":
                churn_top_features,

            "predicted_90_day_revenue":
                clv_top_features
        },

        "churn_visualization_summary":
            churn_summary,

        "clv_visualization_summary":
            clv_summary,

        "dashboard_integration": {

            "explainability_intelligence_page":
                True,

            "customer_360_explainability_tab":
                True,

            "global_explanations":
                True,

            "customer_level_explanations":
                True,

            "precomputed_explanation_lookup":
                True
        },

        "interpretation_rules": {

            "churn_positive_shap":
                (
                    "Pushes the model churn prediction "
                    "upward relative to the SHAP baseline."
                ),

            "churn_negative_shap":
                (
                    "Pushes the model churn prediction "
                    "downward relative to the SHAP baseline."
                ),

            "clv_positive_shap":
                (
                    "Pushes predicted 90-day revenue "
                    "upward relative to the SHAP baseline."
                ),

            "clv_negative_shap":
                (
                    "Pushes predicted 90-day revenue "
                    "downward relative to the SHAP baseline."
                )
        },

        "limitations": [

            (
                "SHAP explains the behavior of the trained "
                "models and does not establish causality."
            ),

            (
                "Customer-level explanations are currently "
                "precomputed for a sample rather than the "
                "entire customer population."
            ),

            (
                "Raw SHAP magnitudes from churn and revenue "
                "models should not be directly compared "
                "because the model outputs operate on "
                "different scales."
            ),

            (
                "The CLV module predicts fixed-horizon "
                "future 90-day revenue and should not be "
                "described as literal lifetime revenue."
            ),

            (
                "Recommendation scores are unrelated to "
                "SHAP probability interpretation and remain "
                "ranking signals rather than calibrated "
                "purchase probabilities."
            )
        ],

        "validation_summary": {

            "all_expected_artifacts_exist":
                all_files_exist,

            "core_explanation_datasets_valid":
                core_data_valid,

            "overall_status":
                overall_status
        }
    }

    # ========================================================
    # SAVE REPORT
    # ========================================================

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print(
        f"Overall Status: {overall_status}"
    )

    print(
        "\nChurn explanation coverage:"
    )

    print(
        (
            f"{churn_coverage['explained_customers']} / "
            f"{churn_coverage['total_customers']} "
            f"customers "
            f"({churn_coverage['coverage_percentage']:.2f}%)"
        )
    )

    print(
        "\nCLV explanation coverage:"
    )

    print(
        (
            f"{clv_coverage['explained_customers']} / "
            f"{clv_coverage['total_customers']} "
            f"customers "
            f"({clv_coverage['coverage_percentage']:.2f}%)"
        )
    )

    print(
        "\nCustomers explained by both models:"
    )

    print(
        common_customers[
            "count"
        ]
    )

    print(
        "\nTop Churn Features:"
    )

    for feature in churn_top_features:

        print(
            f"- {feature['Feature']}: "
            f"{feature['Mean Absolute SHAP']:.6f}"
        )

    print(
        "\nTop Revenue Features:"
    )

    for feature in clv_top_features:

        print(
            f"- {feature['Feature']}: "
            f"{feature['Mean Absolute SHAP']:.6f}"
        )

    print(
        "\nFinal report saved:"
    )

    print(
        REPORT_PATH
    )

    if overall_status == "PASS":

        print(
            "\nPHASE 6 VALIDATION PASSED."
        )

    else:

        print(
            "\nSome explainability artifacts "
            "require review."
        )


if __name__ == "__main__":

    main()