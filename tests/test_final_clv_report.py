import json
import os

import pandas as pd

from src.evaluation.final_clv_report import (
    FinalCLVReport
)


def test_final_clv_report(
    tmp_path
):

    metadata = {

        "model": {
            "name":
                "Gradient Boosting",

            "target_strategy":
                "Raw",

            "artifact_path":
                "models/clv/clv_model.pkl",

            "best_parameters": {
                "n_estimators":
                    200,

                "learning_rate":
                    0.05,

                "max_depth":
                    3
            }
        },

        "selection": {
            "criterion":
                "Lowest cross-validated MAE",

            "best_cv_mae":
                350.50,

            "final_test_used_for_selection":
                False
        },

        "features": [
            "Recency",
            "Frequency",
            "Monetary",
            "TotalItems",
            "AverageOrderValue",
            "Tenure"
        ],

        "target": {
            "name":
                "FutureRevenue",

            "prediction_horizon_days":
                90
        },

        "final_test": {
            "customer_count":
                1000,

            "mae":
                370.25,

            "rmse":
                1800.50,

            "r2":
                0.58,

            "negative_prediction_count":
                0,

            "negative_prediction_percentage":
                0.0,

            "actual_mean":
                850.0,

            "predicted_mean":
                825.0,

            "actual_median":
                250.0,

            "predicted_median":
                275.0
        },

        "artifacts": {
            "feature_importance":
                (
                    "artifacts/clv/final/"
                    "clv_feature_importance.csv"
                ),

            "test_predictions":
                (
                    "artifacts/clv/final/"
                    "clv_final_test_predictions.csv"
                )
        }
    }

    feature_importance = pd.DataFrame(
        {
            "Feature": [
                "Monetary",
                "Frequency",
                "Recency",
                "TotalItems",
                "Tenure",
                "AverageOrderValue"
            ],

            "Importance": [
                0.35,
                0.25,
                0.18,
                0.10,
                0.07,
                0.05
            ],

            "ImportancePercentage": [
                35.0,
                25.0,
                18.0,
                10.0,
                7.0,
                5.0
            ]
        }
    )

    value_bands = {
        "method":
            "Training target quantiles",

        "prediction_horizon_days":
            90,

        "low_upper_bound":
            250.0,

        "medium_upper_bound":
            900.0,

        "definitions": {
            "Low":
                "Predicted revenue <= 250",

            "Medium":
                (
                    "Predicted revenue > 250 "
                    "and <= 900"
                ),

            "High":
                "Predicted revenue > 900"
        }
    }

    output_path = os.path.join(
        tmp_path,
        "clv_model_report.json"
    )

    generator = (
        FinalCLVReport()
    )

    report = (
        generator.generate_report(
            metadata=
                metadata,

            feature_importance=
                feature_importance,

            value_bands=
                value_bands,

            output_path=
                output_path
        )
    )

    assert report is not None

    assert os.path.exists(
        output_path
    )

    assert (
        report[
            "module"
        ][
            "prediction_type"
        ]
        ==
        "Regression"
    )

    assert (
        report[
            "target_definition"
        ][
            "prediction_horizon_days"
        ]
        ==
        90
    )

    assert (
        report[
            "selected_model"
        ][
            "algorithm"
        ]
        ==
        "Gradient Boosting"
    )

    assert (
        report[
            "selected_model"
        ][
            "target_strategy"
        ]
        ==
        "Raw"
    )

    assert (
        report[
            "final_test_performance"
        ][
            "mae"
        ]
        ==
        370.25
    )

    assert (
        len(
            report[
                "feature_importance"
            ]
        )
        ==
        6
    )

    assert (
        report[
            "value_bands"
        ][
            "low_upper_bound"
        ]
        ==
        250.0
    )

    with open(
        output_path,
        "r",
        encoding="utf-8"
    ) as file:

        saved_report = (
            json.load(
                file
            )
        )

    assert (
        saved_report[
            "methodology"
        ][
            "validation"
        ][
            "final_test_used_for_selection"
        ]
        is False
    )