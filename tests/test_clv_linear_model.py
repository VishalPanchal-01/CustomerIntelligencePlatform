import pandas as pd

from sklearn.pipeline import Pipeline

from src.training.clv_linear_model import (
    CLVLinearRegressionModel
)


def test_clv_linear_model():

    X = pd.DataFrame(
        {
            "Recency": [
                10,
                20,
                30,
                40,
                50
            ],

            "Frequency": [
                10,
                8,
                6,
                4,
                2
            ],

            "Monetary": [
                5000,
                4000,
                3000,
                2000,
                1000
            ],

            "TotalItems": [
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
                500
            ],

            "Tenure": [
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
            2500,
            2000,
            1500,
            1000,
            500
        ]
    )

    trainer = (
        CLVLinearRegressionModel()
    )

    model = trainer.train(
        X,
        y
    )

    assert model is not None

    assert isinstance(
        model,
        Pipeline
    )

    assert (
        "scaler"
        in model.named_steps
    )

    assert (
        "regressor"
        in model.named_steps
    )

    predictions = (
        model.predict(
            X
        )
    )

    assert (
        len(predictions)
        == 5
    )