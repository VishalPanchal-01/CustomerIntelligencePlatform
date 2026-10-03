import numpy as np
import pandas as pd

from sklearn.model_selection import (
    RandomizedSearchCV
)

from sklearn.ensemble import (
    RandomForestRegressor
)

from src.evaluation.final_clv_model_selection import (
    FinalCLVModelSelector
)


def create_search(
    X,
    y,
    n_estimators
):

    model = RandomForestRegressor(
        random_state=42
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions={
            "n_estimators": [
                n_estimators
            ]
        },
        n_iter=1,
        scoring=
            "neg_mean_absolute_error",
        cv=2,
        random_state=42
    )

    search.fit(
        X,
        y
    )

    return search


def test_final_clv_candidate_selection():

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
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1
            ]
        }
    )

    y = pd.Series(
        [
            800,
            700,
            600,
            500,
            400,
            300,
            200,
            100
        ]
    )

    search_a = create_search(
        X,
        y,
        10
    )

    search_b = create_search(
        X,
        y,
        20
    )

    candidates = {

        "candidate_a": {
            "model_name":
                "Random Forest",

            "target_type":
                "Raw",

            "search":
                search_a
        },

        "candidate_b": {
            "model_name":
                "Random Forest",

            "target_type":
                "Raw",

            "search":
                search_b
        }
    }

    selector = (
        FinalCLVModelSelector()
    )

    selected = (
        selector.select_best_candidate(
            candidates
        )
    )

    assert selected is not None

    assert (
        "best_estimator"
        in selected
    )

    assert (
        "best_params"
        in selected
    )

    assert (
        "cv_mae"
        in selected
    )

    expected_best_mae = min(
        -search_a.best_score_,
        -search_b.best_score_
    )

    assert np.isclose(
        selected[
            "cv_mae"
        ],
        expected_best_mae
    )


def test_final_clv_evaluation():

    X_train = pd.DataFrame(
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
                6,
                5,
                4,
                3,
                2,
                1
            ]
        }
    )

    y_train = pd.Series(
        [
            600,
            500,
            400,
            300,
            200,
            100
        ]
    )

    model = RandomForestRegressor(
        n_estimators=20,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    X_test = pd.DataFrame(
        {
            "Recency": [
                15,
                45
            ],

            "Frequency": [
                5,
                2
            ]
        }
    )

    y_test = pd.Series(
        [
            550,
            250
        ]
    )

    selector = (
        FinalCLVModelSelector()
    )

    result = (
        selector.evaluate_final_model(
            model,
            X_test,
            y_test
        )
    )

    assert result is not None

    assert (
        result[
            "mae"
        ]
        >= 0
    )

    assert (
        result[
            "rmse"
        ]
        >= 0
    )

    assert np.isfinite(
        result[
            "r2"
        ]
    )

    assert (
        len(
            result[
                "predictions"
            ]
        )
        == 2
    )

    assert (
        result[
            "predictions"
        ]
        >= 0
    ).all()