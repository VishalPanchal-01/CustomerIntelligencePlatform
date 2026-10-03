import numpy as np
import pandas as pd

from src.evaluation.clv_model_comparison import (
    CLVModelComparison
)


def test_clv_model_comparison():

    X_train = pd.DataFrame(
        {
            "Recency": [
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
                55,
                60,
                65
            ],

            "Frequency": [
                12,
                11,
                10,
                9,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1
            ],

            "Monetary": [
                6000,
                5500,
                5000,
                4500,
                4000,
                3500,
                3000,
                2500,
                2000,
                1500,
                1000,
                500
            ],

            "TotalItems": [
                600,
                550,
                500,
                450,
                400,
                350,
                300,
                250,
                200,
                150,
                100,
                50
            ],

            "AverageOrderValue": [
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500
            ],

            "Tenure": [
                400,
                380,
                360,
                340,
                320,
                300,
                280,
                260,
                240,
                220,
                200,
                180
            ]
        }
    )

    y_raw_train = pd.Series(
        [
            3000,
            2800,
            2600,
            2400,
            2200,
            2000,
            1800,
            1600,
            1400,
            1200,
            1000,
            800
        ]
    )

    y_log_train = pd.Series(
        np.log1p(
            y_raw_train
        )
    )

    X_test = pd.DataFrame(
        {
            "Recency": [
                70,
                80,
                90
            ],

            "Frequency": [
                3,
                2,
                1
            ],

            "Monetary": [
                1200,
                800,
                400
            ],

            "TotalItems": [
                120,
                80,
                40
            ],

            "AverageOrderValue": [
                400,
                400,
                400
            ],

            "Tenure": [
                180,
                160,
                140
            ]
        }
    )

    y_raw_test = pd.Series(
        [
            1100,
            700,
            300
        ]
    )

    comparator = (
        CLVModelComparison()
    )

    result = (
        comparator.compare_models(
            X_train=
                X_train,

            X_test=
                X_test,

            y_raw_train=
                y_raw_train,

            y_raw_test=
                y_raw_test,

            y_log_train=
                y_log_train
        )
    )

    assert result is not None

    assert (
        len(result)
        == 8
    )

    expected_columns = [
        "Model",
        "Target",
        "MAE",
        "RMSE",
        "R2",
        "NegativePredictions",
        "NegativePredictionPercentage"
    ]

    assert (
        result.columns.tolist()
        ==
        expected_columns
    )

    assert (
        result[
            "MAE"
        ]
        >= 0
    ).all()

    assert (
        result[
            "RMSE"
        ]
        >= 0
    ).all()

    assert set(
        result[
            "Target"
        ]
    ) == {
        "Raw",
        "Log"
    }

    assert set(
        result[
            "Model"
        ]
    ) == {
        "Dummy Regressor",
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    }