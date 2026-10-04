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


# ============================================================
# NORMALIZE CUSTOMER ID
# ============================================================

def normalize_customer_id(
    value
) -> str:

    if pd.isna(
        value
    ):
        return ""

    value = str(
        value
    ).strip()

    # --------------------------------------------------------
    # Handle IDs stored as:
    #
    # 17850
    # 17850.0
    # "17850"
    # "17850.0"
    # --------------------------------------------------------

    try:

        numeric_value = float(
            value
        )

        if numeric_value.is_integer():

            return str(
                int(
                    numeric_value
                )
            )

    except (
        ValueError,
        TypeError
    ):
        pass

    return value


# ============================================================
# LOAD DATA
# ============================================================

def load_churn_dataset() -> pd.DataFrame:

    if not os.path.exists(
        DATASET_PATH
    ):

        raise FileNotFoundError(
            f"Churn dataset not found: "
            f"{DATASET_PATH}"
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
            "Customer ID column not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # --------------------------------------------------------
    # Normalize customer IDs
    # --------------------------------------------------------

    df[
        "Customer ID Normalized"
    ] = (
        df[
            "Customer ID"
        ]
        .apply(
            normalize_customer_id
        )
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "========================================"
    )

    print(
        "CUSTOMER CHURN SHAP EXPLANATION"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_churn_dataset()

    print(
        f"\nTotal customers available: "
        f"{df['Customer ID Normalized'].nunique():,}"
    )

    # --------------------------------------------------------
    # Show sample customer IDs
    # --------------------------------------------------------

    sample_ids = (
        df[
            "Customer ID Normalized"
        ]
        .dropna()
        .drop_duplicates()
        .head(
            20
        )
        .tolist()
    )

    print(
        "\nSample Customer IDs:"
    )

    print(
        sample_ids
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    entered_customer_id = input(
        "\nEnter Customer ID: "
    )

    customer_id = (
        normalize_customer_id(
            entered_customer_id
        )
    )

    # --------------------------------------------------------
    # Find customer
    # --------------------------------------------------------

    customer_match = (
        df[
            df[
                "Customer ID Normalized"
            ]
            ==
            customer_id
        ]
        .copy()
    )

    if customer_match.empty:

        print(
            "\nCustomer not found."
        )

        print(
            f"You entered: "
            f"{entered_customer_id}"
        )

        print(
            f"Normalized ID: "
            f"{customer_id}"
        )

        print(
            "\nTry one of these available Customer IDs:"
        )

        print(
            sample_ids
        )

        return

    customer = (
        customer_match
        .head(
            1
        )
        .copy()
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature
        in ChurnSHAPExplainer.FEATURE_COLUMNS
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing churn features: "
            f"{missing_features}"
        )

    X = (
        df[
            ChurnSHAPExplainer.FEATURE_COLUMNS
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Initialize SHAP explainer
    # --------------------------------------------------------

    print(
        "\nBuilding SHAP explainer..."
    )

    explainer = (
        ChurnSHAPExplainer(
            model_path=
                MODEL_PATH,

            background_size=
                200,

            random_state=
                42
        )
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
        "CUSTOMER EXPLANATION"
    )

    print(
        "========================================"
    )

    print(
        f"\nCustomer ID: "
        f"{customer_id}"
    )

    print(
        "\nCustomer Features:"
    )

    print(
        customer[
            ChurnSHAPExplainer.FEATURE_COLUMNS
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nPredicted Churn Probability:"
    )

    print(
        f"{summary['churn_probability'] * 100:.2f}%"
    )

    print(
        "\n"
        "----------------------------------------"
    )

    print(
        "FEATURE CONTRIBUTIONS"
    )

    print(
        "----------------------------------------"
    )

    display_columns = [
        "Feature",
        "Feature Value",
        "SHAP Value",
        "Impact Direction",
        "Impact Rank"
    ]

    print(
        explanation[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Churn drivers
    # --------------------------------------------------------

    print(
        "\n"
        "----------------------------------------"
    )

    print(
        "TOP CHURN DRIVERS"
    )

    print(
        "----------------------------------------"
    )

    churn_drivers = (
        summary[
            "top_churn_drivers"
        ]
    )

    if not churn_drivers:

        print(
            "No positive SHAP drivers found."
        )

    else:

        for index, driver in enumerate(
            churn_drivers,
            start=1
        ):

            print(
                f"{index}. "
                f"{driver['Feature']} "
                f"| Value: "
                f"{driver['Feature Value']:.2f} "
                f"| SHAP: "
                f"{driver['SHAP Value']:+.4f}"
            )

    # --------------------------------------------------------
    # Retention drivers
    # --------------------------------------------------------

    print(
        "\n"
        "----------------------------------------"
    )

    print(
        "TOP RETENTION DRIVERS"
    )

    print(
        "----------------------------------------"
    )

    retention_drivers = (
        summary[
            "top_retention_drivers"
        ]
    )

    if not retention_drivers:

        print(
            "No negative SHAP drivers found."
        )

    else:

        for index, driver in enumerate(
            retention_drivers,
            start=1
        ):

            print(
                f"{index}. "
                f"{driver['Feature']} "
                f"| Value: "
                f"{driver['Feature Value']:.2f} "
                f"| SHAP: "
                f"{driver['SHAP Value']:+.4f}"
            )

    print(
        "\n"
        "========================================"
    )

    print(
        "Explanation completed."
    )

    print(
        "========================================"
    )


if __name__ == "__main__":

    main()