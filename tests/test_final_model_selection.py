import pandas as pd

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from src.evaluation.final_model_selection import (
    FinalModelSelector
)


def test_final_model_selection():

    X_train = pd.DataFrame(
        {
            "Recency": [
                10,
                20,
                30,
                100,
                120,
                150
            ],

            "Frequency": [
                8,
                7,
                6,
                2,
                1,
                1
            ]
        }
    )

    y_train = pd.Series(
        [
            0,
            0,
            0,
            1,
            1,
            1
        ]
    )

    X_test = pd.DataFrame(
        {
            "Recency": [
                15,
                140
            ],

            "Frequency": [
                7,
                1
            ]
        }
    )

    y_test = pd.Series(
        [
            0,
            1
        ]
    )

    logistic = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    random_forest = (
        RandomForestClassifier(
            n_estimators=20,
            random_state=42
        )
    )

    logistic.fit(
        X_train,
        y_train
    )

    random_forest.fit(
        X_train,
        y_train
    )

    models = {
        "Logistic Regression":
            logistic,

        "Random Forest":
            random_forest
    }

    selector = (
        FinalModelSelector()
    )

    (
        model_name,
        best_model,
        comparison
    ) = selector.select_best_model(
        models,
        X_test,
        y_test
    )

    assert model_name in models

    assert best_model is not None

    assert comparison is not None

    assert len(
        comparison
    ) == 2

    assert "F1Score" in (
        comparison.columns
    )

    assert "ROCAUC" in (
        comparison.columns
    )