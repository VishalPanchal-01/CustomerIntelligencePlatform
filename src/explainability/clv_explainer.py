import os

import joblib
import numpy as np
import pandas as pd
import shap

from src.explainability.shap_utils import (
    SHAPUtils
)


class CLVSHAPExplainer:

    # =========================================================
    # CONSTANTS
    # =========================================================

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
        background_size: int = 100,
        random_state: int = 42,
        clip_negative_predictions: bool = True
    ):

        self.model_path = model_path

        self.background_size = (
            background_size
        )

        self.random_state = (
            random_state
        )

        self.clip_negative_predictions = (
            clip_negative_predictions
        )

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
                f"CLV model not found: "
                f"{self.model_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        return self.model


    # =========================================================
    # PREDICTION FUNCTION
    # =========================================================

    def _predict_revenue(
        self,
        X
    ) -> np.ndarray:

        if self.model is None:

            self.load_model()

        if isinstance(
            X,
            np.ndarray
        ):

            X = pd.DataFrame(
                X,
                columns=
                    self.FEATURE_COLUMNS
            )

        predictions = (
            self.model
            .predict(
                X
            )
        )

        predictions = np.asarray(
            predictions,
            dtype=float
        ).reshape(
            -1
        )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # TransformedTargetRegressor.predict()
        # already converts log-target predictions back to
        # original revenue scale.
        #
        # DO NOT apply np.expm1() here.
        # -----------------------------------------------------

        if self.clip_negative_predictions:

            predictions = np.clip(
                predictions,
                a_min=0,
                a_max=None
            )

        return predictions


    # =========================================================
    # FIT SHAP EXPLAINER
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

        masker = (
            shap.maskers.Independent(
                self.background_data
            )
        )

        self.explainer = (
            shap.Explainer(
                self._predict_revenue,
                masker,
                feature_names=
                    self.FEATURE_COLUMNS
            )
        )

        return self.explainer


    # =========================================================
    # EXPLAIN DATA
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

        features = (
            SHAPUtils.validate_features(
                X,
                self.FEATURE_COLUMNS
            )
        )

        return (
            self.explainer(
                features
            )
        )


    # =========================================================
    # EXTRACT REGRESSION VALUES
    # =========================================================

    @staticmethod
    def extract_regression_values(
        shap_values
    ) -> np.ndarray:

        if hasattr(
            shap_values,
            "values"
        ):

            values = (
                shap_values.values
            )

        else:

            values = shap_values

        values = np.asarray(
            values
        )

        if values.ndim == 2:

            return values

        # Some wrappers may produce:
        # rows × features × 1
        if (
            values.ndim == 3
            and
            values.shape[-1] == 1
        ):

            return values[
                :,
                :,
                0
            ]

        raise ValueError(
            "Unsupported CLV SHAP value shape: "
            f"{values.shape}"
        )


    # =========================================================
    # GLOBAL FEATURE IMPORTANCE
    # =========================================================

    def global_feature_importance(
        self,
        X: pd.DataFrame,
        max_samples: int = 300
    ) -> pd.DataFrame:

        if self.explainer is None:

            self.fit_explainer(
                X
            )

        sample = (
            SHAPUtils
            .sample_explanation_rows(
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
            self.extract_regression_values(
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
        ] = np.arange(
            1,
            len(result) + 1
        )

        return result


    # =========================================================
    # EXPLAIN ONE CUSTOMER
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
            self.extract_regression_values(
                shap_values
            )[0]
        )

        prediction = float(
            self._predict_revenue(
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

            "Increases Predicted Revenue",

            np.where(
                result[
                    "SHAP Value"
                ]
                <
                0,

                "Decreases Predicted Revenue",

                "Neutral"
            )
        )

        result[
            "Predicted 90-Day Revenue"
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
        ] = np.arange(
            1,
            len(result) + 1
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

        prediction = float(
            explanation[
                "Predicted 90-Day Revenue"
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
            .sort_values(
                by=
                    "Absolute SHAP",
                ascending=False
            )
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
            .sort_values(
                by=
                    "Absolute SHAP",
                ascending=False
            )
            .head(
                top_n
            )
        )

        return {

            "predicted_90_day_revenue":
                prediction,

            "top_value_increasing_factors":
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

            "top_value_decreasing_factors":
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