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


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Customer selection
    # --------------------------------------------------------

    customer_id = input(
        "Enter Customer ID: "
    ).strip()

    customer_match = (
        df[
            df[
                "Customer ID"
            ]
            .astype(str)
            ==
            customer_id
        ]
    )

    if customer_match.empty:

        print(
            "Customer not found."
        )

        return

    customer = (
        customer_match
        .head(1)
        .copy()
    )

    # --------------------------------------------------------
    # Initialize
    # --------------------------------------------------------

    explainer = CLVSHAPExplainer(
        model_path=
            MODEL_PATH,

        background_size=
            100,

        clip_negative_predictions=
            True
    )

    X = df[
        CLVSHAPExplainer.FEATURE_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # Fit explainer
    # --------------------------------------------------------

    print(
        "\nBuilding CLV SHAP explainer..."
    )

    explainer.fit_explainer(
        X
    )

    # --------------------------------------------------------
    # Explain customer
    # --------------------------------------------------------

    explanation = (
        explainer
        .explain_customer(
            customer
        )
    )

    summary = (
        explainer
        .explain_customer_summary(
            customer,
            top_n=3
        )
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\n"
        "========================================"
    )

    print(
        f"Customer ID: {customer_id}"
    )

    print(
        "========================================"
    )

    print(
        "\nPredicted 90-Day Revenue:"
    )

    print(
        f"{summary['predicted_90_day_revenue']:,.2f}"
    )

    print(
        "\nFeature Contributions:"
    )

    print(
        explanation[
            [
                "Feature",
                "Feature Value",
                "SHAP Value",
                "Impact Direction"
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nTop Value-Increasing Factors:"
    )

    increasing = (
        summary[
            "top_value_increasing_factors"
        ]
    )

    if not increasing:

        print(
            "- None"
        )

    else:

        for driver in increasing:

            print(
                f"- {driver['Feature']} "
                f"(value={driver['Feature Value']}, "
                f"SHAP={driver['SHAP Value']:.4f})"
            )

    print(
        "\nTop Value-Decreasing Factors:"
    )

    decreasing = (
        summary[
            "top_value_decreasing_factors"
        ]
    )

    if not decreasing:

        print(
            "- None"
        )

    else:

        for driver in decreasing:

            print(
                f"- {driver['Feature']} "
                f"(value={driver['Feature Value']}, "
                f"SHAP={driver['SHAP Value']:.4f})"
            )


if __name__ == "__main__":

    main()