import os

import numpy as np
import pandas as pd
import shap

from src.explainability.churn_shap_visualization import (
    ChurnSHAPVisualization
)


# ============================================================
# TEST IMPORTANCE DATA
# ============================================================

def create_importance_data():

    return pd.DataFrame(
        {
            "Feature": [
                "Recency",
                "Frequency",
                "Monetary",
                "TotalItems",
                "AverageOrderValue",
                "Tenure"
            ],

            "Mean Absolute SHAP": [
                0.30,
                0.25,
                0.18,
                0.12,
                0.08,
                0.05
            ]
        }
    )


# ============================================================
# TEST EXPLANATION DATA
# ============================================================

def create_explanation_data():

    values = [
        0.30,
        0.20,
        -0.15,
        0.08,
        -0.05,
        -0.04
    ]

    return pd.DataFrame(
        {
            "Feature": [
                "Recency",
                "Frequency",
                "Monetary",
                "TotalItems",
                "AverageOrderValue",
                "Tenure"
            ],

            "Feature Value": [
                120,
                2,
                400,
                20,
                200,
                300
            ],

            "SHAP Value":
                values,

            "Absolute SHAP":
                np.abs(
                    values
                )
        }
    )


# ============================================================
# GLOBAL IMPORTANCE PLOT
# ============================================================

def test_global_importance_plot(
    tmp_path
):

    visualizer = (
        ChurnSHAPVisualization()
    )

    output_path = os.path.join(
        tmp_path,
        "global.png"
    )

    visualizer.save_global_importance_plot(
        create_importance_data(),
        output_path
    )

    assert os.path.exists(
        output_path
    )

    assert (
        os.path.getsize(
            output_path
        )
        >
        0
    )


# ============================================================
# DRIVER PLOT
# ============================================================

def test_customer_driver_plot(
    tmp_path
):

    visualizer = (
        ChurnSHAPVisualization()
    )

    output_path = os.path.join(
        tmp_path,
        "driver.png"
    )

    visualizer.save_customer_driver_plot(
        create_explanation_data(),
        output_path
    )

    assert os.path.exists(
        output_path
    )

    assert (
        os.path.getsize(
            output_path
        )
        >
        0
    )


# ============================================================
# DASHBOARD EXPLANATION DATA
# ============================================================

def test_dashboard_explanation_data():

    visualizer = (
        ChurnSHAPVisualization()
    )

    X = pd.DataFrame(
        {
            "Recency": [
                10,
                100
            ],

            "Frequency": [
                15,
                2
            ],

            "Monetary": [
                4000,
                300
            ]
        }
    )

    explanation = shap.Explanation(
        values=np.array(
            [
                [
                    -0.20,
                    -0.10,
                    -0.05
                ],

                [
                    0.30,
                    0.20,
                    0.10
                ]
            ]
        ),

        base_values=np.array(
            [
                0.5,
                0.5
            ]
        ),

        data=X.values,

        feature_names=X.columns.tolist()
    )

    result = (
        visualizer
        .build_dashboard_explanation_data(
            customer_ids=[
                101,
                102
            ],

            X=X,

            shap_values=
                explanation
        )
    )

    assert (
        len(result)
        ==
        6
    )

    assert (
        "Customer ID"
        in result.columns
    )

    assert (
        "SHAP Value"
        in result.columns
    )

    assert (
        "Impact Direction"
        in result.columns
    )

    assert (
        "Impact Rank"
        in result.columns
    )

    customer_101 = (
        result[
            result[
                "Customer ID"
            ]
            ==
            101
        ]
    )

    assert (
        len(
            customer_101
        )
        ==
        3
    )


# ============================================================
# DRIVER SUMMARY
# ============================================================

def test_driver_summary():

    visualizer = (
        ChurnSHAPVisualization()
    )

    result = (
        visualizer
        .build_customer_driver_summary(
            create_explanation_data(),
            top_n=2
        )
    )

    assert (
        len(
            result[
                "top_churn_drivers"
            ]
        )
        ==
        2
    )

    assert (
        len(
            result[
                "top_retention_drivers"
            ]
        )
        ==
        2
    )

    assert (
        result[
            "top_churn_drivers"
        ][0][
            "Feature"
        ]
        ==
        "Recency"
    )


# ============================================================
# WATERFALL
# ============================================================

def test_waterfall_plot(
    tmp_path
):

    visualizer = (
        ChurnSHAPVisualization()
    )

    explanation = shap.Explanation(
        values=np.array(
            [
                0.3,
                -0.2,
                0.1
            ]
        ),

        base_values=0.4,

        data=np.array(
            [
                100,
                5,
                2000
            ]
        ),

        feature_names=[
            "Recency",
            "Frequency",
            "Monetary"
        ]
    )

    output_path = os.path.join(
        tmp_path,
        "waterfall.png"
    )

    visualizer.save_customer_waterfall_plot(
        explanation,
        output_path
    )

    assert os.path.exists(
        output_path
    )

    assert (
        os.path.getsize(
            output_path
        )
        >
        0
    )
    