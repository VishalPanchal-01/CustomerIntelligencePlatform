import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor

from src.evaluation.clv_evaluation import (
    CLVModelEvaluation
)


def test_clv_raw_evaluation():

    X_train = pd.DataFrame(
        {
            "Feature": [
                1,
                2,
                3,
                4
            ]
        }
    )

    y_train = pd.Series(
        [
            100,
            200,
            300,
            400
        ]
    )

    X_test = pd.DataFrame(
        {
            "Feature": [
                5,
                6
            ]
        }
    )

    y_test = pd.Series(
        [
            250,
            350
        ]
    )

    model = DummyRegressor(
        strategy="mean"
    )

    model.fit(
        X_train,
        y_train
    )

    evaluator = (
        CLVModelEvaluation()
    )

    result = (
        evaluator.evaluate(
            model=model,
            X_test=X_test,
            y_test_raw=y_test,
            target_type="raw"
        )
    )

    assert result is not None

    assert (
        "mae"
        in result
    )

    assert (
        "rmse"
        in result
    )

    assert (
        "r2"
        in result
    )

    assert (
        "predictions"
        in result
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


def test_clv_log_evaluation():

    X_train = pd.DataFrame(
        {
            "Feature": [
                1,
                2,
                3,
                4
            ]
        }
    )

    y_train_raw = pd.Series(
        [
            100,
            200,
            300,
            400
        ]
    )

    y_train_log = pd.Series(
        np.log1p(
            y_train_raw
        )
    )

    X_test = pd.DataFrame(
        {
            "Feature": [
                5,
                6
            ]
        }
    )

    y_test_raw = pd.Series(
        [
            250,
            350
        ]
    )

    model = DummyRegressor(
        strategy="mean"
    )

    model.fit(
        X_train,
        y_train_log
    )

    evaluator = (
        CLVModelEvaluation()
    )

    result = (
        evaluator.evaluate(
            model=model,
            X_test=X_test,
            y_test_raw=y_test_raw,
            target_type="log"
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
        result[
            "predictions"
        ]
        >= 0
    ).all()