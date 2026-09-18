import pandas as pd

from src.training.hyperparameter_tuning import (
    ChurnHyperparameterTuner
)


def test_logistic_hyperparameter_tuning():

    X = pd.DataFrame(
        {
            "Recency": [
                10, 20, 30, 40, 50,
                100, 110, 120, 130, 140
            ],

            "Frequency": [
                10, 9, 8, 7, 6,
                5, 4, 3, 2, 1
            ],

            "Monetary": [
                1000, 900, 800, 700, 600,
                500, 400, 300, 200, 100
            ],

            "TotalItems": [
                100, 90, 80, 70, 60,
                50, 40, 30, 20, 10
            ],

            "AverageOrderValue": [
                100, 100, 100, 100, 100,
                100, 100, 100, 100, 100
            ],

            "Tenure": [
                300, 280, 260, 240, 220,
                100, 80, 60, 40, 20
            ]
        }
    )

    y = pd.Series(
        [
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1
        ]
    )

    tuner = (
        ChurnHyperparameterTuner()
    )

    result = (
        tuner
        .tune_logistic_regression(
            X,
            y
        )
    )

    assert result is not None

    assert result.best_estimator_ is not None

    assert result.best_params_ is not None

    assert (
        "classifier__C"
        in result.best_params_
    )