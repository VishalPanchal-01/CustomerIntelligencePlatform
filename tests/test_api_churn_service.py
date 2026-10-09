import numpy as np
import pandas as pd

from sklearn.linear_model import (
    LogisticRegression
)

from api.churn_service import (
    ChurnPredictionService
)


# ============================================================
# TRAIN TEST MODEL
# ============================================================

def create_trained_service():

    X = pd.DataFrame(
        {
            "Recency": [
                5,
                10,
                20,
                50,
                90,
                120,
                150,
                200
            ],

            "Frequency": [
                30,
                25,
                20,
                12,
                5,
                3,
                2,
                1
            ],

            "Monetary": [
                6000,
                5000,
                4000,
                2500,
                1000,
                700,
                400,
                100
            ],

            "TotalItems": [
                500,
                450,
                350,
                220,
                100,
                60,
                30,
                10
            ],

            "AverageOrderValue": [
                200,
                200,
                200,
                208,
                200,
                233,
                200,
                100
            ],

            "Tenure": [
                600,
                550,
                500,
                400,
                300,
                200,
                100,
                50
            ]
        }
    )

    y = np.array(
        [
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1
        ]
    )

    model = LogisticRegression(
        max_iter=2000
    )

    model.fit(
        X,
        y
    )

    service = ChurnPredictionService(
        model_path="unused.pkl"
    )

    service.model = model

    return service


# ============================================================
# VALID FEATURES
# ============================================================

def test_prepare_features():

    service = create_trained_service()

    features = {
        "Recency":
            100,

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
# PROBABILITY
# ============================================================

def test_predict_probability():

    service = create_trained_service()

    features = {
        "Recency":
            120,

        "Frequency":
            3,

        "Monetary":
            700,

        "TotalItems":
            60,

        "AverageOrderValue":
            233,

        "Tenure":
            200
    }

    X = (
        service
        .prepare_features(
            features
        )
    )

    probability = (
        service
        .predict_probability(
            X
        )
    )

    assert (
        0
        <=
        probability
        <=
        1
    )


# ============================================================
# COMPLETE PREDICTION
# ============================================================

def test_churn_prediction():

    service = create_trained_service()

    features = {
        "Recency":
            150,

        "Frequency":
            2,

        "Monetary":
            400,

        "TotalItems":
            30,

        "AverageOrderValue":
            200,

        "Tenure":
            100
    }

    result = (
        service.predict(
            features
        )
    )

    assert (
        "churn_probability"
        in result
    )

    assert (
        "non_churn_probability"
        in result
    )

    assert (
        "predicted_class"
        in result
    )

    assert (
        "prediction_label"
        in result
    )

    assert (
        "churn_risk"
        in result
    )

    assert (
        "features"
        in result
    )

    assert result[
        "predicted_class"
    ] in [
        0,
        1
    ]


# ============================================================
# RISK BANDS
# ============================================================

def test_risk_band():

    service = create_trained_service()

    assert (
        service.risk_band(
            0.10
        )
        ==
        "Low"
    )

    assert (
        service.risk_band(
            0.50
        )
        ==
        "Medium"
    )

    assert (
        service.risk_band(
            0.80
        )
        ==
        "High"
    )


# ============================================================
# NEGATIVE VALUE
# ============================================================

def test_negative_feature_rejected():

    service = create_trained_service()

    features = {
        "Recency":
            -10,

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
# MODEL STATUS
# ============================================================

def test_model_status():

    service = create_trained_service()

    result = (
        service.model_status()
    )

    assert (
        result[
            "loaded"
        ]
        is True
    )

    assert (
        result[
            "feature_count"
        ]
        ==
        6
    )