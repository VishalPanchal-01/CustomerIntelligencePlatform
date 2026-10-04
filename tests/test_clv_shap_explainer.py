import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor
)

from src.explainability.clv_explainer import (
    CLVSHAPExplainer
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
                90,
                120,
                150,
                200,
                15,
                30
            ],

            "Frequency": [
                25,
                20,
                15,
                10,
                5,
                3,
                2,
                1,
                18,
                12
            ],

            "Monetary": [
                6000,
                5000,
                3500,
                2500,
                1200,
                800,
                500,
                200,
                4500,
                3000
            ],

            "TotalItems": [
                500,
                420,
                300,
                220,
                110,
                70,
                40,
                15,
                380,
                250
            ],

            "AverageOrderValue": [
                240,
                250,
                233,
                250,
                240,
                266,
                250,
                200,
                250,
                250
            ],

            "Tenure": [
                600,
                550,
                500,
                450,
                350,
                300,
                200,
                100,
                520,
                470
            ]
        }
    )

    y = np.array(
        [
            4500,
            4000,
            3000,
            2200,
            1200,
            700,
            400,
            100,
            3600,
            2600
        ]
    )

    return X, y


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def test_clv_prediction_function():

    X, y = create_test_data()

    model = RandomForestRegressor(
        n_estimators=20,
        random_state=42
    )

    model.fit(
        X,
        y
    )

    explainer = CLVSHAPExplainer(
        model_path="unused.pkl"
    )

    explainer.model = model

    predictions = (
        explainer
        ._predict_revenue(
            X
        )
    )

    assert (
        len(predictions)
        ==
        len(X)
    )

    assert np.all(
        predictions
        >=
        0
    )


# ============================================================
# FIT EXPLAINER
# ============================================================

def test_fit_clv_explainer():

    X, y = create_test_data()

    model = RandomForestRegressor(
        n_estimators=20,
        random_state=42
    )

    model.fit(
        X,
        y
    )

    explainer = CLVSHAPExplainer(
        model_path="unused.pkl",
        background_size=5
    )

    explainer.model = model

    result = (
        explainer
        .fit_explainer(
            X
        )
    )

    assert result is not None

    assert (
        explainer.background_data
        is not None
    )

    assert (
        len(
            explainer.background_data
        )
        ==
        5
    )


# ============================================================
# CUSTOMER EXPLANATION
# ============================================================

def test_clv_customer_explanation():

    X, y = create_test_data()

    model = RandomForestRegressor(
        n_estimators=20,
        random_state=42
    )

    model.fit(
        X,
        y
    )

    explainer = CLVSHAPExplainer(
        model_path="unused.pkl",
        background_size=5
    )

    explainer.model = model

    explainer.fit_explainer(
        X
    )

    result = (
        explainer
        .explain_customer(
            X.iloc[
                [
                    0
                ]
            ]
        )
    )

    assert (
        len(result)
        ==
        6
    )

    assert (
        "Feature"
        in result.columns
    )

    assert (
        "SHAP Value"
        in result.columns
    )

    assert (
        "Absolute SHAP"
        in result.columns
    )

    assert (
        "Impact Direction"
        in result.columns
    )

    assert (
        "Predicted 90-Day Revenue"
        in result.columns
    )

    assert (
        "Impact Rank"
        in result.columns
    )


# ============================================================
# GLOBAL IMPORTANCE
# ============================================================

def test_clv_global_feature_importance():

    X, y = create_test_data()

    model = RandomForestRegressor(
        n_estimators=20,
        random_state=42
    )

    model.fit(
        X,
        y
    )

    explainer = CLVSHAPExplainer(
        model_path="unused.pkl",
        background_size=5
    )

    explainer.model = model

    result = (
        explainer
        .global_feature_importance(
            X,
            max_samples=10
        )
    )

    assert (
        len(result)
        ==
        6
    )

    assert (
        "Mean Absolute SHAP"
        in result.columns
    )

    assert (
        "Importance Rank"
        in result.columns
    )

    assert (
        result[
            "Mean Absolute SHAP"
        ]
        .ge(0)
        .all()
    )


# ============================================================
# CUSTOMER SUMMARY
# ============================================================

def test_clv_customer_summary():

    X, y = create_test_data()

    model = RandomForestRegressor(
        n_estimators=20,
        random_state=42
    )

    model.fit(
        X,
        y
    )

    explainer = CLVSHAPExplainer(
        model_path="unused.pkl",
        background_size=5
    )

    explainer.model = model

    explainer.fit_explainer(
        X
    )

    result = (
        explainer
        .explain_customer_summary(
            X.iloc[
                [
                    0
                ]
            ],
            top_n=3
        )
    )

    assert (
        "predicted_90_day_revenue"
        in result
    )

    assert (
        "top_value_increasing_factors"
        in result
    )

    assert (
        "top_value_decreasing_factors"
        in result
    )

    assert (
        result[
            "predicted_90_day_revenue"
        ]
        >=
        0
    )


# ============================================================
# NEGATIVE PREDICTION CLIPPING
# ============================================================

def test_negative_prediction_clipping():

    class DummyNegativeModel:

        def predict(
            self,
            X
        ):

            return np.full(
                len(X),
                -100.0
            )

    X, _ = create_test_data()

    explainer = CLVSHAPExplainer(
        model_path="unused.pkl",
        clip_negative_predictions=True
    )

    explainer.model = (
        DummyNegativeModel()
    )

    prediction = (
        explainer
        ._predict_revenue(
            X.iloc[
                [
                    0
                ]
            ]
        )
    )

    assert (
        prediction[0]
        ==
        0.0
    )