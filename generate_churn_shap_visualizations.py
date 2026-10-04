import json
import os

import numpy as np
import pandas as pd

from src.explainability.churn_explainer import (
    ChurnSHAPExplainer
)

from src.explainability.churn_shap_visualization import (
    ChurnSHAPVisualization
)

from src.explainability.shap_utils import (
    SHAPUtils
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "churn",
    "churn_model.pkl"
)

DATASET_PATH = os.path.join(
    "artifacts",
    "churn",
    "customer_churn_dataset.csv"
)

EXPLAINABILITY_DIRECTORY = os.path.join(
    "artifacts",
    "explainability",
    "churn"
)

VISUALIZATION_DIRECTORY = os.path.join(
    EXPLAINABILITY_DIRECTORY,
    "visualizations"
)

GLOBAL_IMPORTANCE_CSV = os.path.join(
    EXPLAINABILITY_DIRECTORY,
    "churn_global_shap_importance.csv"
)

DASHBOARD_DATA_PATH = os.path.join(
    EXPLAINABILITY_DIRECTORY,
    "churn_dashboard_shap_values.csv"
)

GLOBAL_IMPORTANCE_PNG = os.path.join(
    VISUALIZATION_DIRECTORY,
    "churn_global_feature_importance.png"
)

BEESWARM_PNG = os.path.join(
    VISUALIZATION_DIRECTORY,
    "churn_shap_beeswarm.png"
)

CUSTOMER_DRIVER_PNG = os.path.join(
    VISUALIZATION_DIRECTORY,
    "example_customer_churn_drivers.png"
)

CUSTOMER_WATERFALL_PNG = os.path.join(
    VISUALIZATION_DIRECTORY,
    "example_customer_churn_waterfall.png"
)

SUMMARY_JSON = os.path.join(
    EXPLAINABILITY_DIRECTORY,
    "churn_visualization_summary.json"
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
        "CHURN SHAP VISUALIZATION"
    )

    print(
        "========================================"
        "\n"
    )

    # ========================================================
    # DIRECTORIES
    # ========================================================

    os.makedirs(
        VISUALIZATION_DIRECTORY,
        exist_ok=True
    )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print(
        "Loading churn dataset..."
    )

    df = pd.read_csv(
        DATASET_PATH
    )

    # --------------------------------------------------------
    # Backward compatibility
    # --------------------------------------------------------

    if (
        "CustomerID"
        in df.columns
        and
        "Customer ID"
        not in df.columns
    ):

        df = df.rename(
            columns={
                "CustomerID":
                    "Customer ID"
            }
        )

    if (
        "Customer ID"
        not in df.columns
    ):

        raise ValueError(
            "Customer ID column not found."
        )

    # ========================================================
    # FEATURES
    # ========================================================

    X = df[
        ChurnSHAPExplainer.FEATURE_COLUMNS
    ].copy()

    # ========================================================
    # INITIALIZE EXPLAINER
    # ========================================================

    print(
        "Building churn SHAP explainer..."
    )

    explainer = ChurnSHAPExplainer(
        model_path=
            MODEL_PATH,

        background_size=
            200,

        random_state=
            42
    )

    explainer.fit_explainer(
        X
    )

    # ========================================================
    # VISUALIZER
    # ========================================================

    visualizer = (
        ChurnSHAPVisualization()
    )

    # ========================================================
    # GLOBAL IMPORTANCE
    # ========================================================

    print(
        "Generating global feature importance..."
    )

    global_importance = (
        explainer
        .global_feature_importance(
            X,
            max_samples=300
        )
    )

    global_importance.to_csv(
        GLOBAL_IMPORTANCE_CSV,
        index=False
    )

    visualizer.save_global_importance_plot(
        global_importance,
        GLOBAL_IMPORTANCE_PNG
    )

    # ========================================================
    # SHAP SAMPLE
    # ========================================================

    sample_size = min(
        300,
        len(df)
    )

    sample_df = (
        df
        .sample(
            n=sample_size,
            random_state=42
        )
        .copy()
    )

    sample_X = (
        sample_df[
            ChurnSHAPExplainer.FEATURE_COLUMNS
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    sample_customer_ids = (
        sample_df[
            "Customer ID"
        ]
        .reset_index(
            drop=True
        )
        .tolist()
    )

    print(
        "Calculating SHAP values for "
        f"{sample_size} customers..."
    )

    sample_shap_values = (
        explainer.explain(
            sample_X
        )
    )

    # ========================================================
    # BEESWARM
    # ========================================================

    print(
        "Saving SHAP beeswarm..."
    )

    visualizer.save_beeswarm_plot(
        sample_shap_values,
        BEESWARM_PNG,
        max_display=6
    )

    # ========================================================
    # DASHBOARD DATA
    # ========================================================

    print(
        "Generating dashboard-ready SHAP data..."
    )

    dashboard_data = (
        visualizer
        .build_dashboard_explanation_data(
            customer_ids=
                sample_customer_ids,

            X=
                sample_X,

            shap_values=
                sample_shap_values
        )
    )

    dashboard_data.to_csv(
        DASHBOARD_DATA_PATH,
        index=False
    )

    # ========================================================
    # EXAMPLE CUSTOMER
    #
    # Select highest predicted churn customer from sample.
    # ========================================================

    print(
        "Selecting example high-risk customer..."
    )

    probabilities = (
        explainer
        ._positive_class_probability(
            sample_X
        )
    )

    highest_risk_position = int(
        np.argmax(
            probabilities
        )
    )

    example_customer_id = (
        sample_customer_ids[
            highest_risk_position
        ]
    )

    example_customer = (
        sample_X.iloc[
            [
                highest_risk_position
            ]
        ]
    )

    example_shap = (
        explainer.explain(
            example_customer
        )
    )

    example_explanation = (
        explainer
        .explain_customer(
            example_customer
        )
    )

    # ========================================================
    # CUSTOMER DRIVER CHART
    # ========================================================

    print(
        "Saving example customer driver chart..."
    )

    visualizer.save_customer_driver_plot(
        explanation_df=
            example_explanation,

        output_path=
            CUSTOMER_DRIVER_PNG,

        top_n=
            6
    )

    # ========================================================
    # CUSTOMER WATERFALL
    # ========================================================

    print(
        "Saving example customer waterfall..."
    )

    visualizer.save_customer_waterfall_plot(
        shap_explanation=
            example_shap,

        output_path=
            CUSTOMER_WATERFALL_PNG,

        max_display=
            6
    )

    # ========================================================
    # DRIVER SUMMARY
    # ========================================================

    driver_summary = (
        visualizer
        .build_customer_driver_summary(
            example_explanation,
            top_n=3
        )
    )

    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {

        "module":
            "Customer Churn Prediction",

        "explanation_method":
            "SHAP",

        "sample_size":
            sample_size,

        "example_customer_id":
            (
                str(
                    example_customer_id
                )
            ),

        "example_churn_probability":
            float(
                probabilities[
                    highest_risk_position
                ]
            ),

        "top_global_features":
            (
                global_importance
                .head(
                    6
                )
                .to_dict(
                    orient="records"
                )
            ),

        "top_churn_drivers":
            driver_summary[
                "top_churn_drivers"
            ],

        "top_retention_drivers":
            driver_summary[
                "top_retention_drivers"
            ],

        "artifacts": {

            "global_importance":
                GLOBAL_IMPORTANCE_PNG,

            "beeswarm":
                BEESWARM_PNG,

            "customer_driver":
                CUSTOMER_DRIVER_PNG,

            "customer_waterfall":
                CUSTOMER_WATERFALL_PNG,

            "dashboard_shap_values":
                DASHBOARD_DATA_PATH
        },

        "interpretation_note":
            (
                "Positive SHAP values push the churn "
                "prediction upward relative to the model "
                "baseline. Negative values push it downward. "
                "SHAP explains model behavior and does not "
                "establish causality."
            )
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "\nGlobal SHAP Importance:"
    )

    print(
        global_importance.to_string(
            index=False
        )
    )

    print(
        "\nExample Customer:"
    )

    print(
        example_customer_id
    )

    print(
        "\nExample Churn Probability:"
    )

    print(
        f"{probabilities[highest_risk_position] * 100:.2f}%"
    )

    print(
        "\nSaved artifacts:"
    )

    print(
        GLOBAL_IMPORTANCE_PNG
    )

    print(
        BEESWARM_PNG
    )

    print(
        CUSTOMER_DRIVER_PNG
    )

    print(
        CUSTOMER_WATERFALL_PNG
    )

    print(
        DASHBOARD_DATA_PATH
    )

    print(
        SUMMARY_JSON
    )

    print(
        "\nChurn SHAP visualization completed."
    )


if __name__ == "__main__":

    main()