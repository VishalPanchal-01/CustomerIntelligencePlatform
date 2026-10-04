import os

import pandas as pd
import streamlit as st


CUSTOMER_ID = "Customer ID"


@st.cache_data
def load_customer_intelligence(
    file_path: str
) -> pd.DataFrame:

    """
    Loads the unified customer intelligence dataset.

    Streamlit caching prevents the CSV from being
    loaded again on every widget interaction.
    """

    if not os.path.exists(
        file_path
    ):

        raise FileNotFoundError(
            f"Dashboard dataset not found: "
            f"{file_path}"
        )

    df = pd.read_csv(
        file_path
    )

    # ---------------------------------------------------------
    # Backward compatibility
    # ---------------------------------------------------------

    if (
        "CustomerID" in df.columns
        and
        CUSTOMER_ID not in df.columns
    ):

        df = df.rename(
            columns={
                "CustomerID":
                    CUSTOMER_ID
            }
        )

    if CUSTOMER_ID not in df.columns:

        raise ValueError(
            "Customer ID column not found "
            "in dashboard dataset."
        )

    # ---------------------------------------------------------
    # Remove completely empty Customer IDs
    # ---------------------------------------------------------

    df = df[
        df[
            CUSTOMER_ID
        ].notna()
    ].copy()

    # ---------------------------------------------------------
    # Remove duplicate customer rows
    # ---------------------------------------------------------

    df = (
        df
        .drop_duplicates(
            subset=[
                CUSTOMER_ID
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return df


# =============================================================
# GLOBAL FILTERING
# =============================================================

def filter_dashboard_data(
    df: pd.DataFrame,
    segments=None,
    churn_risks=None,
    clv_bands=None,
    customer_search=None
) -> pd.DataFrame:

    """
    Applies global dashboard filters.
    """

    filtered = (
        df.copy()
    )

    # =========================================================
    # CUSTOMER SEGMENT
    # =========================================================

    if (
        segments
        and
        "Customer Segment"
        in filtered.columns
    ):

        filtered = filtered[
            filtered[
                "Customer Segment"
            ].isin(
                segments
            )
        ]

    # =========================================================
    # CHURN RISK
    # =========================================================

    if (
        churn_risks
        and
        "Churn Risk"
        in filtered.columns
    ):

        filtered = filtered[
            filtered[
                "Churn Risk"
            ].isin(
                churn_risks
            )
        ]

    # =========================================================
    # CLV BAND
    # =========================================================

    if (
        clv_bands
        and
        "CLV Value Band"
        in filtered.columns
    ):

        filtered = filtered[
            filtered[
                "CLV Value Band"
            ].isin(
                clv_bands
            )
        ]

    # =========================================================
    # CUSTOMER SEARCH
    # =========================================================

    if (
        customer_search is not None
        and
        str(
            customer_search
        ).strip()
        !=
        ""
    ):

        search_value = (
            str(
                customer_search
            )
            .strip()
            .lower()
        )

        filtered = filtered[
            filtered[
                CUSTOMER_ID
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                search_value,
                na=False
            )
        ]

    return (
        filtered
        .reset_index(
            drop=True
        )
    )


# =============================================================
# OVERVIEW KPI CALCULATION
# =============================================================

def calculate_overview_metrics(
    df: pd.DataFrame
) -> dict:

    """
    Calculates reusable executive dashboard KPIs.
    """

    total_customers = int(
        df[
            CUSTOMER_ID
        ]
        .nunique()
    )

    # =========================================================
    # HIGH CHURN RISK
    # =========================================================

    if (
        "Churn Risk"
        in df.columns
    ):

        high_risk_customers = int(
            (
                df[
                    "Churn Risk"
                ]
                ==
                "High"
            )
            .sum()
        )

    else:

        high_risk_customers = 0

    # =========================================================
    # HIGH VALUE CUSTOMERS
    # =========================================================

    if (
        "CLV Value Band"
        in df.columns
    ):

        high_value_customers = int(
            (
                df[
                    "CLV Value Band"
                ]
                ==
                "High"
            )
            .sum()
        )

    else:

        high_value_customers = 0

    # =========================================================
    # PREDICTED REVENUE
    # =========================================================

    if (
        "Predicted 90-Day Revenue"
        in df.columns
    ):

        predicted_revenue = (
            pd.to_numeric(
                df[
                    "Predicted 90-Day Revenue"
                ],
                errors="coerce"
            )
        )

        average_predicted_revenue = float(
            predicted_revenue
            .mean()
        )

        total_predicted_revenue = float(
            predicted_revenue
            .sum()
        )

    else:

        average_predicted_revenue = 0.0
        total_predicted_revenue = 0.0

    # =========================================================
    # AVERAGE CHURN PROBABILITY
    # =========================================================

    if (
        "Churn Probability"
        in df.columns
    ):

        churn_probability = (
            pd.to_numeric(
                df[
                    "Churn Probability"
                ],
                errors="coerce"
            )
        )

        average_churn_probability = float(
            churn_probability
            .mean()
        )

    else:

        average_churn_probability = 0.0

    return {

        "total_customers":
            total_customers,

        "high_risk_customers":
            high_risk_customers,

        "high_value_customers":
            high_value_customers,

        "average_predicted_revenue":
            average_predicted_revenue,

        "total_predicted_revenue":
            total_predicted_revenue,

        "average_churn_probability":
            average_churn_probability
    }


# =============================================================
# SAFE CATEGORY EXTRACTION
# =============================================================

def get_unique_values(
    df: pd.DataFrame,
    column: str
) -> list:

    """
    Returns sorted non-null values for sidebar filters.
    """

    if column not in df.columns:

        return []

    values = (
        df[
            column
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return sorted(
        values
    )