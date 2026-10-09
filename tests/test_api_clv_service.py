import json
import os

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor
)

from api.clv_service import (
    CLVPredictionService
)


# ============================================================
# TEST DATA
# ============================================================

def create_test_data():

    X = pd.DataFrame(
        {
            "Recency": [
                5,
                10,
                20,
                50,
                100,
                150
            ],

            "Frequency": [
                25,
                20,
                15,
                8,
                3,
                1
            ],

            "Monetary": [
                6000,
                5000,
                3500,
                2000,
                800,
                200
            ],

            "TotalItems": [
                500,
                420,
                300,
                170,
                60,
                15
            ],

            "AverageOrderValue": [
                240,
                250,
                233,
                250,
                266,
                200
            ],

            "Tenure": [
                600,
                550,
                500,
                400,
                250,
                100
            ]
        }
    )

    y = np.array(
        [
            4500,
            3900,
            3000,
            1800,
            700,
            150
        ]
    )

    return X, y


# ============================================================
# SERVICE
# ============================================================

def create_service():

    X, y = create_test_data()

    model = RandomForestRegressor(
        n_estimators=20,
        random_state=42
    )

    model.fit(
        X,
        y
    )

    service = CLVPredictionService(
        model_path="unused.pkl",
        value_bands_path="unused.json"
    )

    service.model = model

    service.value_bands = {
        "low_threshold":
            1000.0,

        "high_threshold":
            3000.0
    }

    return service


# ============================================================
# PREPARE FEATURES
# ============================================================

def test_prepare_features():

    service = create_service()

    features = {
        "Recency":
            50,

        "Frequency":
            8,

        "Monetary":
            2000,

        "TotalItems":
            170,

        "AverageOrderValue":
            250,

        "Tenure":
            400
    }

    X = (
        service
        .prepare_features(
            features
        )
    )

    assert (
        list(
            X.columns
        )
        ==
        service.FEATURE_COLUMNS
    )

    assert (
        len(X)
        ==
        1
    )


# ============================================================
# PREDICTION
# ============================================================

def test_predict_clv():

    service = create_service()

    features = {
        "Recency":
            20,

        "Frequency":
            15,

        "Monetary":
            3500,

        "TotalItems":
            300,

        "AverageOrderValue":
            233,

        "Tenure":
            500
    }

    result = (
        service.predict(
            features
        )
    )

    assert (
        result[
            "predicted_90_day_revenue"
        ]
        >=
        0
    )

    assert (
        result[
            "clv_value_band"
        ]
        in [
            "Low",
            "Medium",
            "High"
        ]
    )


# ============================================================
# VALUE BAND
# ============================================================

def test_value_band():

    service = create_service()

    assert (
        service.value_band(
            500
        )
        ==
        "Low"
    )

    assert (
        service.value_band(
            2000
        )
        ==
        "Medium"
    )

    assert (
        service.value_band(
            4000
        )
        ==
        "High"
    )


# ============================================================
# NEGATIVE VALUES
# ============================================================

def test_negative_feature_rejected():

    service = create_service()

    features = {
        "Recency":
            -1,

        "Frequency":
            5,

        "Monetary":
            1000,

        "TotalItems":
            100,

        "AverageOrderValue":
            200,

        "Tenure":
            300
    }

    try:

        service.prepare_features(
            features
        )

        assert False

    except ValueError as error:

        assert (
            "cannot be negative"
            in str(
                error
            )
        )


# ============================================================
# VALUE BAND FILE NORMALIZATION
# ============================================================

def test_value_band_metadata_q33_q66():

    result = (
        CLVPredictionService
        ._normalize_value_bands(
            {
                "q33":
                    1000,

                "q66":
                    3000
            }
        )
    )

    assert (
        result[
            "low_threshold"
        ]
        ==
        1000
    )

    assert (
        result[
            "high_threshold"
        ]
        ==
        3000
    )


# ============================================================
# LOAD VALUE BANDS
# ============================================================

def test_load_value_bands(
    tmp_path
):

    path = os.path.join(
        tmp_path,
        "bands.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "low_threshold":
                    1000,

                "high_threshold":
                    3000
            },
            file
        )

    service = CLVPredictionService(
        model_path="unused.pkl",
        value_bands_path=path
    )

    result = (
        service.load_value_bands()
    )

    assert (
        result[
            "low_threshold"
        ]
        ==
        1000.0
    )

    assert (
        result[
            "high_threshold"
        ]
        ==
        3000.0
    )