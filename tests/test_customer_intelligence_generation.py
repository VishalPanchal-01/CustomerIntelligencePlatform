import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.dashboard.customer_intelligence_generation import (
    CustomerIntelligenceGenerator
)


# =============================================================
# TEST CUSTOMER DATA
# =============================================================

def create_customer_data():

    return pd.DataFrame(
        {
            # Old name intentionally used
            # for backward compatibility test.
            "CustomerID": [
                101,
                102,
                103,
                104,
                105,
                106
            ],

            "Recency": [
                10,
                120,
                30,
                180,
                20,
                90
            ],

            "Frequency": [
                12,
                2,
                8,
                1,
                15,
                3
            ],

            "Monetary": [
                5000,
                300,
                2500,
                100,
                7000,
                700
            ],

            "TotalItems": [
                300,
                20,
                150,
                5,
                500,
                40
            ],

            "AverageOrderValue": [
                416.67,
                150.00,
                312.50,
                100.00,
                466.67,
                233.33
            ],

            "Tenure": [
                500,
                200,
                350,
                90,
                600,
                220
            ]
        }
    )


# =============================================================
# TEST CHURN MODEL
# =============================================================

def create_churn_model():

    customers = (
        create_customer_data()
    )

    features = [
        "Recency",
        "Frequency",
        "Monetary",
        "TotalItems",
        "AverageOrderValue",
        "Tenure"
    ]

    X = customers[
        features
    ]

    y = np.array(
        [
            0,
            1,
            0,
            1,
            0,
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

    return model


# =============================================================
# MOCK CLV PREDICTOR
# =============================================================

class MockCLVPredictor:

    def predict_batch(
        self,
        customers
    ):

        result = (
            customers.copy()
        )

        result[
            "PredictedFutureRevenue"
        ] = [
            2000,
            100,
            900,
            50,
            3000,
            400
        ]

        result[
            "CLVValueBand"
        ] = [
            "High",
            "Low",
            "Medium",
            "Low",
            "High",
            "Medium"
        ]

        result[
            "PredictionHorizonDays"
        ] = 90

        result[
            "WasNegativeBeforeClipping"
        ] = False

        return result


# =============================================================
# MOCK RECOMMENDATION PREDICTOR
# =============================================================

class MockRecommendationPredictor:

    def recommend_batch(
        self,
        customer_ids,
        top_k=5,
        mode="next_purchase"
    ):

        rows = []

        for customer_id in customer_ids:

            for rank in range(
                1,
                top_k + 1
            ):

                rows.append(
                    {
                        "CustomerID":
                            customer_id,

                        "KnownCustomer":
                            True,

                        "Mode":
                            mode,

                        "RecommendationSource":
                            "Test Recommender",

                        "StockCode":
                            f"P{rank}",

                        "Description":
                            f"Product {rank}",

                        "Score":
                            1.0
                            /
                            rank,

                        "Rank":
                            rank
                    }
                )

        return pd.DataFrame(
            rows
        )


# =============================================================
# NORMALIZATION TEST
# =============================================================

def test_customer_id_normalization():

    generator = (
        CustomerIntelligenceGenerator()
    )

    data = (
        create_customer_data()
    )

    result = (
        generator.normalize_customer_id(
            data
        )
    )

    assert (
        "Customer ID"
        in result.columns
    )

    assert (
        "CustomerID"
        not in result.columns
    )


# =============================================================
# FEATURE PREPARATION
# =============================================================

def test_prepare_customer_features():

    generator = (
        CustomerIntelligenceGenerator()
    )

    data = (
        create_customer_data()
    )

    (
        normalized,
        features,
        customer_ids
    ) = (
        generator.prepare_customer_features(
            data
        )
    )

    assert (
        "Customer ID"
        in normalized.columns
    )

    assert (
        features.columns.tolist()
        ==
        generator.FEATURES
    )

    assert (
        len(customer_ids)
        ==
        6
    )


# =============================================================
# CHURN
# =============================================================

def test_generate_churn_predictions():

    generator = (
        CustomerIntelligenceGenerator()
    )

    customers = (
        create_customer_data()
    )

    model = (
        create_churn_model()
    )

    result = (
        generator.generate_churn_predictions(
            customer_df=
                customers,

            churn_model=
                model
        )
    )

    assert (
        len(result)
        ==
        6
    )

    assert (
        "Customer ID"
        in result.columns
    )

    assert (
        "CustomerID"
        not in result.columns
    )

    assert (
        "Churn Probability"
        in result.columns
    )

    assert (
        "Churn Prediction"
        in result.columns
    )

    assert (
        "Churn Risk"
        in result.columns
    )

    assert (
        (
            result[
                "Churn Probability"
            ]
            >=
            0
        )
        &
        (
            result[
                "Churn Probability"
            ]
            <=
            1
        )
    ).all()

    assert set(
        result[
            "Churn Risk"
        ]
    ).issubset(
        {
            "Low",
            "Medium",
            "High"
        }
    )


# =============================================================
# CLV
# =============================================================

def test_generate_clv_predictions():

    generator = (
        CustomerIntelligenceGenerator()
    )

    customers = (
        create_customer_data()
    )

    predictor = (
        MockCLVPredictor()
    )

    result = (
        generator.generate_clv_predictions(
            customer_df=
                customers,

            clv_predictor=
                predictor
        )
    )

    assert (
        len(result)
        ==
        6
    )

    assert (
        "Customer ID"
        in result.columns
    )

    assert (
        "Predicted 90-Day Revenue"
        in result.columns
    )

    assert (
        "CLV Value Band"
        in result.columns
    )

    assert (
        "CLV Prediction Horizon Days"
        in result.columns
    )


# =============================================================
# RECOMMENDATIONS
# =============================================================

def test_generate_recommendations():

    generator = (
        CustomerIntelligenceGenerator()
    )

    customers = (
        create_customer_data()
    )

    predictor = (
        MockRecommendationPredictor()
    )

    result = (
        generator.generate_recommendations(
            customer_df=
                customers,

            recommendation_predictor=
                predictor,

            top_k=
                3,

            mode=
                "next_purchase"
        )
    )

    assert (
        result[
            "Customer ID"
        ]
        .nunique()
        ==
        6
    )

    assert (
        len(result)
        ==
        18
    )

    assert (
        "Customer ID"
        in result.columns
    )

    assert (
        "Stock Code"
        in result.columns
    )

    assert (
        "Recommendation Score"
        in result.columns
    )

    assert (
        "Recommendation Rank"
        in result.columns
    )

    assert (
        "Recommendation Source"
        in result.columns
    )


# =============================================================
# GENERATION SUMMARY
# =============================================================

def test_generation_summary():

    generator = (
        CustomerIntelligenceGenerator()
    )

    customers = (
        create_customer_data()
    )

    churn = pd.DataFrame(
        {
            "Customer ID": [
                101,
                102,
                103,
                104,
                105,
                106
            ]
        }
    )

    clv = pd.DataFrame(
        {
            "Customer ID": [
                101,
                102,
                103,
                104,
                105,
                106
            ]
        }
    )

    recommendations = pd.DataFrame(
        {
            "Customer ID": [
                101,
                102,
                103,
                104,
                105,
                106
            ]
        }
    )

    summary = (
        generator.create_generation_summary(
            customer_df=
                customers,

            churn_df=
                churn,

            clv_df=
                clv,

            recommendation_df=
                recommendations
        )
    )

    assert (
        summary[
            "total_customers"
        ]
        ==
        6
    )

    assert (
        summary[
            "churn"
        ][
            "coverage_percentage"
        ]
        ==
        100.0
    )

    assert (
        summary[
            "clv"
        ][
            "coverage_percentage"
        ]
        ==
        100.0
    )

    assert (
        summary[
            "recommendation"
        ][
            "coverage_percentage"
        ]
        ==
        100.0
    )