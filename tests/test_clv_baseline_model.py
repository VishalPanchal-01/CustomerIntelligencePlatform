import pandas as pd

from sklearn.dummy import DummyRegressor

from src.training.clv_baseline_model import (
    CLVBaselineModel
)


def test_clv_baseline_model():

    X = pd.DataFrame(
        {
            "Recency": [
                10,
                20,
                30,
                40
            ],

            "Frequency": [
                5,
                4,
                3,
                2
            ]
        }
    )

    y = pd.Series(
        [
            100,
            200,
            300,
            400
        ]
    )

    trainer = (
        CLVBaselineModel(
            strategy="median"
        )
    )

    model = trainer.train(
        X,
        y
    )

    assert model is not None

    assert isinstance(
        model,
        DummyRegressor
    )

    predictions = model.predict(
        X
    )

    assert (
        len(predictions)
        == 4
    )

    assert (
        predictions
        ==
        250
    ).all()