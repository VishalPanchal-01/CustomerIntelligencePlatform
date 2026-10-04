import os

import numpy as np
import pandas as pd
import shap

from src.explainability.clv_shap_visualization import (
    CLVSHAPVisualization
)


# ============================================================
# IMPORTANCE TEST DATA
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
                80.0,
                250.0,
                500.0,
                180.0,
                90.0,
                60.0
            ]
        }
    )


# ============================================================
# CUSTOMER EXPLANATION DATA
# ============================================================

def create_explanation_data():

    values = [
        -120.0,
        300.0,
        700.0,
        210.0,
        -80.0,
        50.0
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
                25,
                18,
                5200,
                400,
                289,
                550
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
# GLOBAL IMPORTANCE IMAGE
# ============================================================

def test_global_importance_plot(
    tmp_path
):

    visualizer = (
        CLVSHAPVisualization()
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
# CUSTOMER DRIVER IMAGE
# ============================================================

def test_customer_driver_plot(
    tmp_path
):

    visualizer = (
        CLVSHAPVisualization()
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
# DASHBOARD DATA
# ============================================================

def test_dashboard_explanation_data():

    visualizer = (
        CLVSHAPVisualization()
    )

    X = pd.DataFrame(
        {
            "Recency": [
                10,
                100
            ],

            "Frequency": [
                20,
                2
            ],

            "Monetary": [
                5000,
                500
            ]
        }
    )

    explanation = shap.Explanation(
        values=np.array(
            [
                [
                    50.0,
                    200.0,
                    600.0
                ],

                [
                    -100.0,
                    -150.0,
                    -250.0
                ]
            ]
        ),

        base_values=np.array(
            [
                1000.0,
                1000.0
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

            X=
                X,

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
        "Feature"
        in result.columns
    )

    assert (
        "SHAP Value"
        in result.columns
    )

    assert (
        "Absolute SHAP"
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


# ============================================================
# DRIVER SUMMARY
# ============================================================

def test_driver_summary():

    visualizer = (
        CLVSHAPVisualization()
    )

    summary = (
        visualizer
        .build_customer_driver_summary(
            create_explanation_data(),
            top_n=2
        )
    )

    assert (
        len(
            summary[
                "top_value_increasing_factors"
            ]
        )
        ==
        2
    )

    assert (
        len(
            summary[
                "top_value_decreasing_factors"
            ]
        )
        ==
        2
    )

    assert (
        summary[
            "top_value_increasing_factors"
        ][0][
            "Feature"
        ]
        ==
        "Monetary"
    )


# ============================================================
# WATERFALL
# ============================================================

def test_waterfall_plot(
    tmp_path
):

    visualizer = (
        CLVSHAPVisualization()
    )

    explanation = shap.Explanation(
        values=np.array(
            [
                500.0,
                -200.0,
                100.0
            ]
        ),

        base_values=
            1200.0,

        data=np.array(
            [
                5000,
                20,
                10
            ]
        ),

        feature_names=[
            "Monetary",
            "Frequency",
            "Recency"
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