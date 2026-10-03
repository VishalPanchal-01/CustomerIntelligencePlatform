import os

import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor
)

from src.utils.clv_model_persistence import (
    CLVModelPersistence
)


def test_clv_model_persistence(
    tmp_path
):

    X = pd.DataFrame(
        {
            "Recency": [
                10,
                20,
                30,
                40
            ],

            "Frequency": [
                4,
                3,
                2,
                1
            ]
        }
    )

    y = pd.Series(
        [
            400,
            300,
            200,
            100
        ]
    )

    model = RandomForestRegressor(
        n_estimators=10,
        random_state=42
    )

    model.fit(
        X,
        y
    )

    file_path = os.path.join(
        tmp_path,
        "clv_model.pkl"
    )

    persistence = (
        CLVModelPersistence()
    )

    persistence.save_model(
        model,
        file_path
    )

    assert os.path.exists(
        file_path
    )

    loaded_model = (
        persistence.load_model(
            file_path
        )
    )

    original_predictions = (
        model.predict(
            X
        )
    )

    loaded_predictions = (
        loaded_model.predict(
            X
        )
    )

    assert (
        original_predictions
        ==
        loaded_predictions
    ).all()