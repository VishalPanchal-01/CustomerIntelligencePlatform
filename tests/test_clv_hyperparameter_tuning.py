import pandas as pd

from sklearn.model_selection import (
    RandomizedSearchCV
)

from src.training.clv_hyperparameter_tuning import (
    CLVHyperparameterTuner
)


def create_clv_tuning_test_data():

    X = pd.DataFrame(
        {
            "Recency": [
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
                55,
                60,
                65,
                70,
                75,
                80,
                85,
                90,
                95,
                100,
                105
            ],

            "Frequency": [
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
                9,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1
            ],

            "Monetary": [
                10000,
                9500,
                9000,
                8500,
                8000,
                7500,
                7000,
                6500,
                6000,
                5500,
                5000,
                4500,
                4000,
                3500,
                3000,
                2500,
                2000,
                1500,
                1000,
                500
            ],

            "TotalItems": [
                1000,
                950,
                900,
                850,
                800,
                750,
                700,
                650,
                600,
                550,
                500,
                450,
                400,
                350,
                300,
                250,
                200,
                150,
                100,
                50
            ],

            "AverageOrderValue": [
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
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
                485,
                470,
                455,
                440,
                425,
                410,
                395,
                380,
                365,
                350,
                335,
                320,
                305,
                290,
                275,
                260,
                245,
                230,
                215
            ]
        }
    )

    y = pd.Series(
        [
            5000,
            4750,
            4500,
            4250,
            4000,
            3750,
            3500,
            3250,
            3000,
            2750,
            2500,
            2250,
            2000,
            1750,
            1500,
            1250,
            1000,
            750,
            500,
            250
        ],
        name="FutureRevenue"
    )

    return X, y


def test_random_forest_raw_tuning():

    X, y = (
        create_clv_tuning_test_data()
    )

    tuner = (
        CLVHyperparameterTuner(
            n_splits=2,
            random_state=42
        )
    )

    search = (
        tuner.tune_random_forest_raw(
            X_train=X,
            y_train=y,
            n_iter=2
        )
    )

    assert search is not None

    assert isinstance(
        search,
        RandomizedSearchCV
    )

    assert (
        search.best_estimator_
        is not None
    )

    assert isinstance(
        search.best_params_,
        dict
    )

    assert (
        search.best_score_
        <= 0
    )


def test_gradient_boosting_log_tuning():

    X, y = (
        create_clv_tuning_test_data()
    )

    tuner = (
        CLVHyperparameterTuner(
            n_splits=2,
            random_state=42
        )
    )

    search = (
        tuner.tune_gradient_boosting_log(
            X_train=X,
            y_train_raw=y,
            n_iter=2
        )
    )

    assert search is not None

    assert isinstance(
        search,
        RandomizedSearchCV
    )

    assert (
        search.best_estimator_
        is not None
    )

    assert isinstance(
        search.best_params_,
        dict
    )

    assert (
        search.best_score_
        <= 0
    )