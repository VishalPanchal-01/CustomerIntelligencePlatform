import json
import os

import pandas as pd

from src.explainability.churn_explainer import (
    ChurnSHAPExplainer
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

OUTPUT_DIRECTORY = os.path.join(
    "artifacts",
    "explainability",
    "churn"
)

GLOBAL_IMPORTANCE_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "churn_global_shap_importance.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIRECTORY,
    "churn_shap_summary.json"
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
        "CHURN SHAP EXPLAINABILITY"
    )

    print(
        "========================================"
        "\n"
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load churn dataset
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Initialize explainer
    # --------------------------------------------------------

    explainer = ChurnSHAPExplainer(
        model_path=
            MODEL_PATH,

        background_size=
            200,

        random_state=
            42
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X = df[
        ChurnSHAPExplainer.FEATURE_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # Fit SHAP explainer
    # --------------------------------------------------------

    print(
        "Building SHAP explainer..."
    )

    explainer.fit_explainer(
        X
    )

    # --------------------------------------------------------
    # Global feature importance
    # --------------------------------------------------------

    print(
        "Generating global SHAP importance..."
    )

    global_importance = (
        explainer
        .global_feature_importance(
            X,
            max_samples=500
        )
    )

    global_importance.to_csv(
        GLOBAL_IMPORTANCE_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    top_features = (
        global_importance
        .head(6)
        .to_dict(
            orient="records"
        )
    )

    summary = {

        "module":
            "Customer Churn Prediction",

        "explanation_method":
            "SHAP",

        "target":
            "Churn",

        "feature_columns":
            ChurnSHAPExplainer.FEATURE_COLUMNS,

        "background_sample_size":
            min(
                200,
                len(X)
            ),

        "explained_sample_size":
            min(
                500,
                len(X)
            ),

        "top_global_features":
            top_features,

        "interpretation":
            (
                "Positive SHAP values increase the "
                "model churn prediction. Negative SHAP "
                "values decrease the model churn prediction."
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
    # Output
    # --------------------------------------------------------

    print(
        "\nGlobal SHAP importance:"
    )

    print(
        global_importance
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
        "\nSHAP generation completed."
    )


if __name__ == "__main__":

    main()