import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor
)

from src.training.clv_gradient_boosting_model import (
    CLVGradientBoostingModel
)


def test_clv_gradient_boosting_model():

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

    trainer = (
        CLVGradientBoostingModel(
            n_estimators=20,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        )
    )

    model = (
        trainer.train(
            X,
            y
        )
    )

    assert model is not None

    assert isinstance(
        model,
        GradientBoostingRegressor
    )

    assert (
        model.n_estimators
        == 20
    )

    predictions = (
        model.predict(
            X
        )
    )

    assert (
        len(predictions)
        == 8
    )