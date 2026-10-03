import pandas as pd

from src.training.clv_linear_model import (
    CLVLinearRegressionModel
)

from src.training.clv_random_forest_model import (
    CLVRandomForestModel
)

from src.analysis.clv_model_analysis import (
    CLVModelAnalysis
)


def create_test_data():

    X = pd.DataFrame(
        {
            "Recency": [
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80
            ],

            "Frequency": [
                16,
                14,
                12,
                10,
                8,
                6,
                4,
                2
            ],

            "Monetary": [
                8000,
                7000,
                6000,
                5000,
                4000,
                3000,
                2000,
                1000
            ],

            "TotalItems": [
                800,
                700,
                600,
                500,
                400,
                300,
                200,
                100
            ],

            "AverageOrderValue": [
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
                500,
                450,
                400,
                350,
                300,
                250,
                200,
                150
            ]
        }
    )

    y = pd.Series(
        [
            4000,
            3500,
            3000,
            2500,
            2000,
            1500,
            1000,
            500
        ]
    )

    return X, y


def test_clv_linear_coefficients():

    X, y = (
        create_test_data()
    )

    trainer = (
        CLVLinearRegressionModel()
    )

    model = trainer.train(
        X,
        y
    )

    analyzer = (
        CLVModelAnalysis()
    )

    result = (
        analyzer
        .analyze_linear_coefficients(
            model,
            X.columns.tolist()
        )
    )

    assert result is not None

    assert (
        len(result)
        == 6
    )

    assert (
        "Feature"
        in result.columns
    )

    assert (
        "Coefficient"
        in result.columns
    )

    assert (
        "AbsoluteCoefficient"
        in result.columns
    )


def test_clv_random_forest_importance():

    X, y = (
        create_test_data()
    )

    trainer = (
        CLVRandomForestModel(
            n_estimators=20,
            random_state=42
        )
    )

    model = trainer.train(
        X,
        y
    )

    analyzer = (
        CLVModelAnalysis()
    )

    result = (
        analyzer
        .analyze_random_forest_importance(
            model,
            X.columns.tolist()
        )
    )

    assert result is not None

    assert (
        len(result)
        == 6
    )

    assert (
        "Feature"
        in result.columns
    )

    assert (
        "Importance"
        in result.columns
    )

    assert (
        "ImportancePercentage"
        in result.columns
    )

    assert abs(
        result[
            "Importance"
        ].sum()
        -
        1.0
    ) < 0.01