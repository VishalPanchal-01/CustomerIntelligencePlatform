import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.explainability.churn_explainer import (
    ChurnSHAPExplainer
)

from src.explainability.shap_utils import (
    SHAPUtils
)


# ============================================================
# TEST DATA
# ============================================================

def create_test_data():

    X = pd.DataFrame(
        {
            "Recency": [
                10,
                50,
                100,
                5,
                150,
                20,
                90,
                15
            ],

            "Frequency": [
                20,
                10,
                2,
                25,
                1,
                18,
                3,
                22
            ],

            "Monetary": [
                4000,
                2000,
                300,
                5000,
                100,
                3500,
                500,
                4200
            ],

            "TotalItems": [
                300,
                150,
                20,
                400,
                5,
                250,
                30,
                320
            ],

            "AverageOrderValue": [
                200,
                200,
                150,
                200,
                100,
                194,
                166,
                190
            ],

            "Tenure": [
                500,
                300,
                100,
                600,
                50,
                450,
                120,
                550
            ]
        }
    )

    y = np.array(
        [
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            0
        ]
    )

    return X, y


# ============================================================
# VALIDATE FEATURES
# ============================================================

def test_validate_features():

    X, _ = create_test_data()

    result = (
        SHAPUtils.validate_features(
            X,
            ChurnSHAPExplainer.FEATURE_COLUMNS
        )
    )

    assert list(
        result.columns
    ) == ChurnSHAPExplainer.FEATURE_COLUMNS

    assert (
        len(result)
        ==
        len(X)
    )


# ============================================================
# BACKGROUND SAMPLE
# ============================================================

def test_background_sample():

    X, _ = create_test_data()

    result = (
        SHAPUtils.sample_background(
            X,
            max_samples=4,
            random_state=42
        )
    )

    assert (
        len(result)
        ==
        4
    )


# ============================================================
# MODEL PROBABILITY
# ============================================================

def test_positive_class_probability():

    X, y = create_test_data()

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X,
        y
    )

    explainer = ChurnSHAPExplainer(
        model_path="unused.pkl"
    )

    explainer.model = model

    probabilities = (
        explainer
        ._positive_class_probability(
            X
        )
    )

    assert (
        len(probabilities)
        ==
        len(X)
    )

    assert np.all(
        probabilities
        >=
        0
    )

    assert np.all(
        probabilities
        <=
        1
    )


# ============================================================
# CUSTOMER EXPLANATION
# ============================================================

def test_customer_explanation():

    X, y = create_test_data()

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X,
        y
    )

    explainer = ChurnSHAPExplainer(
        model_path="unused.pkl",
        background_size=6
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
        "Impact Direction"
        in result.columns
    )

    assert (
        "Impact Rank"
        in result.columns
    )


# ============================================================
# GLOBAL IMPORTANCE
# ============================================================

def test_global_feature_importance():

    X, y = create_test_data()

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X,
        y
    )

    explainer = ChurnSHAPExplainer(
        model_path="unused.pkl",
        background_size=6
    )

    explainer.model = model

    result = (
        explainer
        .global_feature_importance(
            X,
            max_samples=8
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