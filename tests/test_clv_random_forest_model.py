import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor
)

from src.training.clv_random_forest_model import (
    CLVRandomForestModel
)


def test_clv_random_forest_model():

    X = pd.DataFrame(
        {
            "Recency": [
                10,
                20,
                30,
                40,
                50,
                60
            ],

            "Frequency": [
                12,
                10,
                8,
                6,
                4,
                2
            ],

            "Monetary": [
                6000,
                5000,
                4000,
                3000,
                2000,
                1000
            ],

            "TotalItems": [
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
                500
            ],

            "Tenure": [
                450,
                400,
                350,
                300,
                250,
                200
            ]
        }
    )

    y = pd.Series(
        [
            3000,
            2500,
            2000,
            1500,
            1000,
            500
        ]
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

    assert model is not None

    assert isinstance(
        model,
        RandomForestRegressor
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
        == 6
    )

    assert (
        predictions >= 0
    ).all()