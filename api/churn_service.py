import os

import joblib
import numpy as np
import pandas as pd


class ChurnPredictionService:

    # =========================================================
    # MODEL FEATURES
    # ============================================================

    FEATURE_COLUMNS = [
        "Recency",
        "Frequency",
        "Monetary",
        "TotalItems",
        "AverageOrderValue",
        "Tenure"
    ]


    # =========================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        model_path: str,
        decision_threshold: float = 0.50,
        low_risk_threshold: float = 0.30,
        high_risk_threshold: float = 0.70
    ):

        if not (
            0
            <=
            low_risk_threshold
            <=
            decision_threshold
            <=
            high_risk_threshold
            <=
            1
        ):

            raise ValueError(
                "Thresholds must satisfy:\n"
                "0 <= low_risk_threshold "
                "<= decision_threshold "
                "<= high_risk_threshold <= 1"
            )

        self.model_path = model_path

        self.decision_threshold = (
            decision_threshold
        )

        self.low_risk_threshold = (
            low_risk_threshold
        )

        self.high_risk_threshold = (
            high_risk_threshold
        )

        self.model = None


    # =========================================================
    # MODEL AVAILABILITY
    # ============================================================

    def model_available(
        self
    ) -> bool:

        return os.path.exists(
            self.model_path
        )


    # =========================================================
    # LOAD MODEL
    # ============================================================

    def load_model(
        self
    ):

        if not self.model_available():

            raise FileNotFoundError(
                "Persisted churn model not found: "
                f"{self.model_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        return self.model


    # =========================================================
    # ENSURE MODEL LOADED
    # ============================================================

    def _ensure_model_loaded(
        self
    ) -> None:

        if self.model is None:

            self.load_model()


    # =========================================================
    # PREPARE FEATURES
    # ============================================================

    def prepare_features(
        self,
        features: dict
    ) -> pd.DataFrame:

        missing_features = [
            feature
            for feature in self.FEATURE_COLUMNS
            if feature not in features
        ]

        if missing_features:

            raise ValueError(
                "Missing churn features: "
                f"{missing_features}"
            )

        data = {
            feature:
                features[
                    feature
                ]
            for feature in self.FEATURE_COLUMNS
        }

        X = pd.DataFrame(
            [
                data
            ]
        )

        # -----------------------------------------------------
        # Convert all features to numeric
        # -----------------------------------------------------

        for feature in self.FEATURE_COLUMNS:

            X[
                feature
            ] = pd.to_numeric(
                X[
                    feature
                ],
                errors="coerce"
            )

        # -----------------------------------------------------
        # Missing / invalid values
        # -----------------------------------------------------

        invalid_columns = (
            X
            .columns[
                X.isnull().any()
            ]
            .tolist()
        )

        if invalid_columns:

            raise ValueError(
                "Invalid or non-numeric values "
                "found in features: "
                f"{invalid_columns}"
            )

        # -----------------------------------------------------
        # Infinite values
        # -----------------------------------------------------

        numeric_values = (
            X[
                self.FEATURE_COLUMNS
            ]
            .to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            numeric_values
        ).all():

            raise ValueError(
                "Feature values must be finite."
            )

        # -----------------------------------------------------
        # Negative values
        # -----------------------------------------------------

        negative_features = [
            feature
            for feature in self.FEATURE_COLUMNS
            if float(
                X.iloc[
                    0
                ][
                    feature
                ]
            )
            <
            0
        ]

        if negative_features:

            raise ValueError(
                "Churn features cannot be negative: "
                f"{negative_features}"
            )

        return X


    # =========================================================
    # POSITIVE CLASS PROBABILITY
    # ============================================================

    def predict_probability(
        self,
        X: pd.DataFrame
    ) -> float:

        self._ensure_model_loaded()

        if not hasattr(
            self.model,
            "predict_proba"
        ):

            raise TypeError(
                "Persisted churn model does not "
                "support predict_proba()."
            )

        probabilities = (
            self.model
            .predict_proba(
                X[
                    self.FEATURE_COLUMNS
                ]
            )
        )

        probabilities = np.asarray(
            probabilities,
            dtype=float
        )

        if (
            probabilities.ndim != 2
            or
            probabilities.shape[
                0
            ]
            !=
            1
            or
            probabilities.shape[
                1
            ]
            <
            2
        ):

            raise ValueError(
                "Unexpected predict_proba() "
                f"output shape: {probabilities.shape}"
            )

        probability = float(
            probabilities[
                0,
                1
            ]
        )

        probability = float(
            np.clip(
                probability,
                0.0,
                1.0
            )
        )

        return probability


    # =========================================================
    # RISK BAND
    # ============================================================

    def risk_band(
        self,
        probability: float
    ) -> str:

        if (
            probability
            <
            self.low_risk_threshold
        ):

            return "Low"

        if (
            probability
            <
            self.high_risk_threshold
        ):

            return "Medium"

        return "High"


    # =========================================================
    # PREDICTION LABEL
    # ============================================================

    def prediction_label(
        self,
        predicted_class: int
    ) -> str:

        if predicted_class == 1:

            return "Churn"

        return "Non-Churn"


    # =========================================================
    # PREDICT
    # ============================================================

    def predict(
        self,
        features: dict
    ) -> dict:

        X = self.prepare_features(
            features
        )

        churn_probability = (
            self.predict_probability(
                X
            )
        )

        non_churn_probability = (
            1.0
            -
            churn_probability
        )

        predicted_class = int(
            churn_probability
            >=
            self.decision_threshold
        )

        result_features = {
            feature:
                float(
                    X.iloc[
                        0
                    ][
                        feature
                    ]
                )
            for feature in self.FEATURE_COLUMNS
        }

        return {

            "churn_probability":
                churn_probability,

            "non_churn_probability":
                non_churn_probability,

            "predicted_class":
                predicted_class,

            "prediction_label":
                self.prediction_label(
                    predicted_class
                ),

            "churn_risk":
                self.risk_band(
                    churn_probability
                ),

            "decision_threshold":
                self.decision_threshold,

            "features":
                result_features
        }


    # =========================================================
    # MODEL STATUS
    # ============================================================

    def model_status(
        self
    ) -> dict:

        return {

            "available":
                self.model_available(),

            "loaded":
                self.model is not None,

            "model_path":
                self.model_path,

            "feature_count":
                len(
                    self.FEATURE_COLUMNS
                ),

            "features":
                self.FEATURE_COLUMNS,

            "decision_threshold":
                self.decision_threshold,

            "low_risk_threshold":
                self.low_risk_threshold,

            "high_risk_threshold":
                self.high_risk_threshold
        }