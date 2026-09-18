import os

import pandas as pd

from sklearn.linear_model import (
    LogisticRegression
)

from src.utils.model_persistence import (
    ModelPersistence
)


def test_model_save_and_load(
    tmp_path
):

    X = pd.DataFrame(
        {
            "Recency": [
                10,
                20,
                100,
                120
            ],

            "Frequency": [
                5,
                4,
                2,
                1
            ]
        }
    )

    y = pd.Series(
        [
            0,
            0,
            1,
            1
        ]
    )

    model = LogisticRegression(
        random_state=42
    )

    model.fit(
        X,
        y
    )

    persistence = (
        ModelPersistence()
    )

    model_path = os.path.join(
        tmp_path,
        "test_model.pkl"
    )

    saved_path = (
        persistence.save_model(
            model,
            model_path
        )
    )

    assert os.path.exists(
        saved_path
    )

    loaded_model = (
        persistence.load_model(
            saved_path
        )
    )

    assert loaded_model is not None

    original_predictions = (
        model.predict(X)
    )

    loaded_predictions = (
        loaded_model.predict(X)
    )

    assert (
        original_predictions
        ==
        loaded_predictions
    ).all()