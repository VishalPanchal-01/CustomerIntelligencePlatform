import json
import os

import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor
)

from src.prediction.clv_predictor import (
    CLVPredictor
)


def create_test_model(
    tmp_path
):

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

    model = RandomForestRegressor(
        n_estimators=20,
        random_state=42
    )

    model.fit(
        X,
        y
    )

    model_path = os.path.join(
        tmp_path,
        "clv_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    band_path = os.path.join(
        tmp_path,
        "clv_value_bands.json"
    )

    band_config = {
        "low_upper_bound":
            1000.0,

        "medium_upper_bound":
            2000.0
    }

    with open(
        band_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            band_config,
            file
        )

    return (
        model_path,
        band_path
    )


def test_single_clv_prediction(
    tmp_path
):

    (
        model_path,
        band_path
    ) = create_test_model(
        tmp_path
    )

    predictor = CLVPredictor(
        model_path=
            model_path,

        value_band_path=
            band_path
    )

    customer = {

        "Recency":
            20,

        "Frequency":
            10,

        "Monetary":
            5000,

        "TotalItems":
            500,

        "AverageOrderValue":
            500,

        "Tenure":
            400
    }

    result = (
        predictor.predict(
            customer
        )
    )

    assert result is not None

    assert (
        "predicted_future_revenue"
        in result
    )

    assert (
        "value_band"
        in result
    )

    assert (
        result[
            "prediction_horizon_days"
        ]
        == 90
    )

    assert (
        result[
            "predicted_future_revenue"
        ]
        >= 0
    )

    assert (
        result[
            "value_band"
        ]
        in [
            "Low",
            "Medium",
            "High"
        ]
    )


def test_batch_clv_prediction(
    tmp_path
):

    (
        model_path,
        band_path
    ) = create_test_model(
        tmp_path
    )

    predictor = CLVPredictor(
        model_path=
            model_path,

        value_band_path=
            band_path
    )

    customers = pd.DataFrame(
        {
            "CustomerID": [
                101,
                102
            ],

            "Recency": [
                15,
                50
            ],

            "Frequency": [
                11,
                4
            ],

            "Monetary": [
                5500,
                2000
            ],

            "TotalItems": [
                550,
                200
            ],

            "AverageOrderValue": [
                500,
                500
            ],

            "Tenure": [
                420,
                250
            ]
        }
    )

    result = (
        predictor.predict_batch(
            customers
        )
    )

    assert result is not None

    assert (
        len(result)
        == 2
    )

    assert (
        "CustomerID"
        in result.columns
    )

    assert (
        "PredictedFutureRevenue"
        in result.columns
    )

    assert (
        "CLVValueBand"
        in result.columns
    )

    assert (
        "PredictionHorizonDays"
        in result.columns
    )

    assert (
        result[
            "PredictedFutureRevenue"
        ]
        >= 0
    ).all()

    assert (
        result[
            "PredictionHorizonDays"
        ]
        == 90
    ).all()

import pytest


def test_clv_predictor_missing_feature(
    tmp_path
):

    (
        model_path,
        band_path
    ) = create_test_model(
        tmp_path
    )

    predictor = CLVPredictor(
        model_path=
            model_path,

        value_band_path=
            band_path
    )

    customer = {

        "Recency":
            20,

        "Frequency":
            5,

        "Monetary":
            2000,

        "TotalItems":
            100,

        # AverageOrderValue missing

        "Tenure":
            200
    }

    with pytest.raises(
        Exception
    ):

        predictor.predict(
            customer
        )


def test_clv_predictor_negative_feature(
    tmp_path
):

    (
        model_path,
        band_path
    ) = create_test_model(
        tmp_path
    )

    predictor = CLVPredictor(
        model_path=
            model_path,

        value_band_path=
            band_path
    )

    customer = {

        "Recency":
            -10,

        "Frequency":
            5,

        "Monetary":
            2000,

        "TotalItems":
            100,

        "AverageOrderValue":
            400,

        "Tenure":
            200
    }

    with pytest.raises(
        Exception
    ):

        predictor.predict(
            customer
        )
