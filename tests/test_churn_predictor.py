import os

import pandas as pd

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler
)

from src.prediction.churn_predictor import (
    ChurnPredictor
)

from src.utils.model_persistence import (
    ModelPersistence
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
                100,
                120,
                150
            ],

            "Frequency": [
                10,
                8,
                6,
                2,
                1,
                1
            ],

            "Monetary": [
                1000,
                800,
                600,
                200,
                100,
                50
            ],

            "TotalItems": [
                100,
                80,
                60,
                20,
                10,
                5
            ],

            "AverageOrderValue": [
                100,
                100,
                100,
                100,
                100,
                50
            ],

            "Tenure": [
                300,
                250,
                200,
                50,
                30,
                10
            ]
        }
    )

    y = pd.Series(
        [
            0,
            0,
            0,
            1,
            1,
            1
        ]
    )

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    model.fit(
        X,
        y
    )

    model_path = os.path.join(
        tmp_path,
        "test_churn_model.pkl"
    )

    persistence = ModelPersistence()

    persistence.save_model(
        model,
        model_path
    )

    return model_path


def test_churn_predictor(
    tmp_path
):

    model_path = create_test_model(
        tmp_path
    )

    predictor = ChurnPredictor(
        model_path=model_path
    )

    customer_data = {
        "Recency": 130,
        "Frequency": 1,
        "Monetary": 100,
        "TotalItems": 10,
        "AverageOrderValue": 100,
        "Tenure": 20
    }

    result = predictor.predict(
        customer_data
    )

    assert result is not None

    assert (
        result["prediction"]
        in [0, 1]
    )

    assert (
        result[
            "churn_probability"
        ]
        is not None
    )

    assert (
        0
        <=
        result[
            "churn_probability"
        ]
        <=
        1
    )

    assert (
        result["risk_level"]
        in [
            "Low",
            "Medium",
            "High"
        ]
    )


def test_churn_predictor_feature_order(
    tmp_path
):

    model_path = create_test_model(
        tmp_path
    )

    predictor = ChurnPredictor(
        model_path=model_path
    )

    customer_data = {
        "Tenure": 100,
        "Monetary": 500,
        "Recency": 50,
        "AverageOrderValue": 100,
        "Frequency": 5,
        "TotalItems": 50
    }

    result = predictor.predict(
        customer_data
    )

    assert result is not None

    assert (
        result["prediction"]
        in [0, 1]
    )


def test_risk_level(
    tmp_path
):

    model_path = create_test_model(
        tmp_path
    )

    predictor = ChurnPredictor(
        model_path=model_path
    )

    assert (
        predictor.get_risk_level(
            0.20
        )
        ==
        "Low"
    )

    assert (
        predictor.get_risk_level(
            0.50
        )
        ==
        "Medium"
    )

    assert (
        predictor.get_risk_level(
            0.80
        )
        ==
        "High"
    )