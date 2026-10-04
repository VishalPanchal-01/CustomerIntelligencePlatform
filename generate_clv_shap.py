import json
import os

import pandas as pd

from src.explainability.clv_explainer import (
    CLVSHAPExplainer
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

OUTPUT_DIRECTORY = os.path.join(
    "artifacts",
    "explainability",
    "clv"
)

GLOBAL_IMPORTANCE_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "clv_global_shap_importance.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "clv_shap_summary.json"
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
        "CLV SHAP EXPLAINABILITY"
    )

    print(
        "========================================"
        "\n"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print(
        "Loading CLV dataset..."
    )

    df = pd.read_csv(
        DATASET_PATH
    )

    # --------------------------------------------------------
    # Customer ID backward compatibility
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

    # --------------------------------------------------------
    # Feature data
    # --------------------------------------------------------

    X = df[
        CLVSHAPExplainer.FEATURE_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # Initialize
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Build SHAP explainer
    # --------------------------------------------------------

    print(
        "Building CLV SHAP explainer..."
    )

    explainer.fit_explainer(
        X
    )

    # --------------------------------------------------------
    # Global importance
    # --------------------------------------------------------

    print(
        "Generating global CLV SHAP importance..."
    )

    global_importance = (
        explainer
        .global_feature_importance(
            X,
            max_samples=300
        )
    )

    global_importance.to_csv(
        GLOBAL_IMPORTANCE_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "module":
            (
                "Customer 90-Day Revenue "
                "Prediction"
            ),

        "explanation_method":
            "SHAP",

        "prediction_target":
            "FutureRevenue",

        "business_label":
            "Predicted 90-Day Revenue",

        "features":
            CLVSHAPExplainer.FEATURE_COLUMNS,

        "background_sample_size":
            min(
                100,
                len(X)
            ),

        "explained_sample_size":
            min(
                300,
                len(X)
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

        "prediction_scale":
            "Original revenue scale",

        "negative_predictions_clipped":
            True,

        "important_note":
            (
                "If the persisted model uses "
                "TransformedTargetRegressor, predict() "
                "already returns predictions on the "
                "original revenue scale. No additional "
                "expm1 transformation is applied."
            ),

        "interpretation":
            (
                "Positive SHAP values push predicted "
                "90-day revenue upward relative to the "
                "SHAP baseline. Negative SHAP values "
                "push the prediction downward. SHAP "
                "explains model behavior and does not "
                "establish causal effects."
            )
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\nGlobal CLV SHAP Importance:\n"
    )

    print(
        global_importance.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        GLOBAL_IMPORTANCE_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        "\nCLV SHAP generation completed."
    )


if __name__ == "__main__":

    main()