import json
import os

import numpy as np
import pandas as pd

from src.explainability.clv_explainer import (
    CLVSHAPExplainer
)

from src.explainability.clv_shap_visualization import (
    CLVSHAPVisualization
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    "models",
    "clv",
    "clv_model.pkl"
)

DATASET_PATH = os.path.join(
    "artifacts",
    "clv",
    "customer_clv_dataset.csv"
)

EXPLAINABILITY_DIRECTORY = os.path.join(
    "artifacts",
    "explainability",
    "clv"
)

VISUALIZATION_DIRECTORY = os.path.join(
    EXPLAINABILITY_DIRECTORY,
    "visualizations"
)

GLOBAL_IMPORTANCE_CSV = os.path.join(
    EXPLAINABILITY_DIRECTORY,
    "clv_global_shap_importance.csv"
)

DASHBOARD_DATA_PATH = os.path.join(
    EXPLAINABILITY_DIRECTORY,
    "clv_dashboard_shap_values.csv"
)

GLOBAL_IMPORTANCE_PNG = os.path.join(
    VISUALIZATION_DIRECTORY,
    "clv_global_feature_importance.png"
)

BEESWARM_PNG = os.path.join(
    VISUALIZATION_DIRECTORY,
    "clv_shap_beeswarm.png"
)

CUSTOMER_DRIVER_PNG = os.path.join(
    VISUALIZATION_DIRECTORY,
    "example_customer_clv_drivers.png"
)

CUSTOMER_WATERFALL_PNG = os.path.join(
    VISUALIZATION_DIRECTORY,
    "example_customer_clv_waterfall.png"
)

SUMMARY_JSON = os.path.join(
    EXPLAINABILITY_DIRECTORY,
    "clv_visualization_summary.json"
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
        "CLV SHAP VISUALIZATION"
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
    # LOAD DATA
    # ========================================================

    print(
        "Loading CLV dataset..."
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
        CLVSHAPExplainer.FEATURE_COLUMNS
    ].copy()

    # ========================================================
    # EXPLAINER
    # ========================================================

    print(
        "Building CLV SHAP explainer..."
    )

    explainer = CLVSHAPExplainer(
        model_path=
            MODEL_PATH,

        background_size=
            100,

        random_state=
            42,

        clip_negative_predictions=
            True
    )

    explainer.fit_explainer(
        X
    )

    # ========================================================
    # VISUALIZER
    # ========================================================

    visualizer = (
        CLVSHAPVisualization()
    )

    # ========================================================
    # GLOBAL IMPORTANCE
    # ========================================================

    print(
        "Generating global CLV importance..."
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
    # SAMPLE DATA
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
            CLVSHAPExplainer.FEATURE_COLUMNS
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

    # ========================================================
    # SHAP VALUES
    # ========================================================

    print(
        f"Calculating CLV SHAP values "
        f"for {sample_size} customers..."
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
        "Saving CLV SHAP beeswarm..."
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
        "Generating dashboard-ready "
        "CLV SHAP data..."
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
    # Choose customer with highest predicted
    # 90-day revenue from sampled population.
    # ========================================================

    predictions = (
        explainer
        ._predict_revenue(
            sample_X
        )
    )

    highest_value_position = int(
        np.argmax(
            predictions
        )
    )

    example_customer_id = (
        sample_customer_ids[
            highest_value_position
        ]
    )

    example_customer = (
        sample_X.iloc[
            [
                highest_value_position
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
        "Saving example customer "
        "revenue driver chart..."
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
    # WATERFALL
    # ========================================================

    print(
        "Saving example customer "
        "CLV waterfall..."
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
            (
                "Customer Predicted "
                "90-Day Revenue"
            ),

        "explanation_method":
            "SHAP",

        "sample_size":
            sample_size,

        "example_customer_id":
            str(
                example_customer_id
            ),

        "example_predicted_90_day_revenue":
            float(
                predictions[
                    highest_value_position
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

        "top_value_increasing_factors":
            driver_summary[
                "top_value_increasing_factors"
            ],

        "top_value_decreasing_factors":
            driver_summary[
                "top_value_decreasing_factors"
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

        "prediction_scale":
            "Original 90-day revenue scale",

        "important_note":
            (
                "The final CLV prediction function "
                "already returns values on the original "
                "revenue scale. No additional expm1 "
                "transformation is applied."
            ),

        "interpretation_note":
            (
                "Positive SHAP values push predicted "
                "90-day revenue upward relative to the "
                "model baseline. Negative SHAP values "
                "push it downward. SHAP explains model "
                "behavior and does not establish causality."
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
    # OUTPUT
    # ========================================================

    print(
        "\nGlobal CLV SHAP Importance:"
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
        "\nPredicted 90-Day Revenue:"
    )

    print(
        f"{predictions[highest_value_position]:,.2f}"
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
        "\nCLV SHAP visualization completed."
    )


if __name__ == "__main__":

    main()