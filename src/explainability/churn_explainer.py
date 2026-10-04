import os

import joblib
import numpy as np
import pandas as pd
import shap

from src.explainability.shap_utils import (
    SHAPUtils
)


class ChurnSHAPExplainer:

    FEATURE_COLUMNS = [
        "Recency",
        "Frequency",
        "Monetary",
        "TotalItems",
        "AverageOrderValue",
        "Tenure"
    ]

    CUSTOMER_ID = "Customer ID"


    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        model_path: str,
        background_size: int = 200,
        random_state: int = 42
    ):

        self.model_path = model_path
        self.background_size = background_size
        self.random_state = random_state

        self.model = None
        self.explainer = None
        self.background_data = None


    # =========================================================
    # LOAD MODEL
    # =========================================================

    def load_model(
        self
    ):

        if not os.path.exists(
            self.model_path
        ):

            raise FileNotFoundError(
                f"Churn model not found: "
                f"{self.model_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        return self.model


    # =========================================================
    # PREDICTION FUNCTION
    # =========================================================

    def _positive_class_probability(
        self,
        X
    ):

        if isinstance(
            X,
            np.ndarray
        ):

            X = pd.DataFrame(
                X,
                columns=
                    self.FEATURE_COLUMNS
            )

        if hasattr(
            self.model,
            "predict_proba"
        ):

            probabilities = (
                self.model
                .predict_proba(
                    X
                )
            )

            return probabilities[
                :,
                1
            ]

        predictions = (
            self.model
            .predict(
                X
            )
        )

        return np.asarray(
            predictions,
            dtype=float
        )


    # =========================================================
    # BUILD EXPLAINER
    # =========================================================

    def fit_explainer(
        self,
        X_background: pd.DataFrame
    ):

        if self.model is None:

            self.load_model()

        X_background = (
            SHAPUtils.validate_features(
                X_background,
                self.FEATURE_COLUMNS
            )
        )

        self.background_data = (
            SHAPUtils.sample_background(
                X_background,
                max_samples=
                    self.background_size,
                random_state=
                    self.random_state
            )
        )

        masker = shap.maskers.Independent(
            self.background_data
        )

        self.explainer = shap.Explainer(
            self._positive_class_probability,
            masker,
            feature_names=
                self.FEATURE_COLUMNS
        )

        return self.explainer


    # =========================================================
    # EXPLAIN
    # =========================================================

    def explain(
        self,
        X: pd.DataFrame
    ):

        if self.explainer is None:

            raise RuntimeError(
                "SHAP explainer is not fitted. "
                "Call fit_explainer() first."
            )

        X = (
            SHAPUtils.validate_features(
                X,
                self.FEATURE_COLUMNS
            )
        )

        return self.explainer(
            X
        )


    # =========================================================
    # GLOBAL FEATURE IMPORTANCE
    # =========================================================

    def global_feature_importance(
        self,
        X: pd.DataFrame,
        max_samples: int = 500
    ) -> pd.DataFrame:

        if self.explainer is None:

            self.fit_explainer(
                X
            )

        sample = (
            SHAPUtils.sample_explanation_rows(
                X,
                max_samples=
                    max_samples,
                random_state=
                    self.random_state
            )
        )

        shap_values = (
            self.explain(
                sample
            )
        )

        values = (
            SHAPUtils.extract_binary_class_values(
                shap_values
            )
        )

        importance = (
            np.abs(
                values
            )
            .mean(
                axis=0
            )
        )

        result = pd.DataFrame(
            {
                "Feature":
                    self.FEATURE_COLUMNS,

                "Mean Absolute SHAP":
                    importance
            }
        )

        result = (
            result
            .sort_values(
                by=
                    "Mean Absolute SHAP",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        result[
            "Importance Rank"
        ] = (
            np.arange(
                1,
                len(result) + 1
            )
        )

        return result


    # =========================================================
    # CUSTOMER EXPLANATION
    # =========================================================

    def explain_customer(
        self,
        customer_row: pd.DataFrame
    ) -> pd.DataFrame:

        if len(
            customer_row
        ) != 1:

            raise ValueError(
                "Customer explanation requires "
                "exactly one row."
            )

        features = (
            SHAPUtils.validate_features(
                customer_row,
                self.FEATURE_COLUMNS
            )
        )

        shap_values = (
            self.explain(
                features
            )
        )

        values = (
            SHAPUtils.extract_binary_class_values(
                shap_values
            )
        )[0]

        prediction = float(
            self._positive_class_probability(
                features
            )[0]
        )

        result = pd.DataFrame(
            {
                "Feature":
                    self.FEATURE_COLUMNS,

                "Feature Value":
                    features.iloc[
                        0
                    ].values,

                "SHAP Value":
                    values
            }
        )

        result[
            "Absolute SHAP"
        ] = (
            result[
                "SHAP Value"
            ]
            .abs()
        )

        result[
            "Impact Direction"
        ] = np.where(
            result[
                "SHAP Value"
            ]
            >
            0,
            "Increases Churn Prediction",
            "Decreases Churn Prediction"
        )

        result[
            "Churn Probability"
        ] = prediction

        result = (
            result
            .sort_values(
                by=
                    "Absolute SHAP",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        result[
            "Impact Rank"
        ] = (
            np.arange(
                1,
                len(result) + 1
            )
        )

        return result


    # =========================================================
    # CUSTOMER SUMMARY
    # =========================================================

    def explain_customer_summary(
        self,
        customer_row: pd.DataFrame,
        top_n: int = 3
    ) -> dict:

        explanation = (
            self.explain_customer(
                customer_row
            )
        )

        probability = float(
            explanation[
                "Churn Probability"
            ]
            .iloc[0]
        )

        increasing = (
            explanation[
                explanation[
                    "SHAP Value"
                ]
                >
                0
            ]
            .head(
                top_n
            )
        )

        decreasing = (
            explanation[
                explanation[
                    "SHAP Value"
                ]
                <
                0
            ]
            .head(
                top_n
            )
        )

        return {

            "churn_probability":
                probability,

            "top_churn_drivers":
                increasing[
                    [
                        "Feature",
                        "Feature Value",
                        "SHAP Value"
                    ]
                ]
                .to_dict(
                    orient="records"
                ),

            "top_retention_drivers":
                decreasing[
                    [
                        "Feature",
                        "Feature Value",
                        "SHAP Value"
                    ]
                ]
                .to_dict(
                    orient="records"
                )
        }