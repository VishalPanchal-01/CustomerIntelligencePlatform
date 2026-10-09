import json
import os

import joblib
import numpy as np
import pandas as pd


class CLVPredictionService:

    FEATURE_COLUMNS = [
        "Recency",
        "Frequency",
        "Monetary",
        "TotalItems",
        "AverageOrderValue",
        "Tenure"
    ]

    def __init__(
        self,
        model_path: str,
        value_bands_path: str,
        clip_negative_predictions: bool = True
    ):

        self.model_path = model_path
        self.value_bands_path = value_bands_path

        self.clip_negative_predictions = (
            clip_negative_predictions
        )

        self.model = None
        self.value_bands = None


    # =========================================================
    # MODEL AVAILABILITY
    # =========================================================

    def model_available(
        self
    ) -> bool:

        return os.path.exists(
            self.model_path
        )


    # =========================================================
    # VALUE BANDS AVAILABILITY
    # =========================================================

    def value_bands_available(
        self
    ) -> bool:

        return os.path.exists(
            self.value_bands_path
        )


    # =========================================================
    # LOAD MODEL
    # =========================================================

    def load_model(
        self
    ):

        if not self.model_available():

            raise FileNotFoundError(
                "Persisted CLV model not found: "
                f"{self.model_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        return self.model


    # =========================================================
    # LOAD VALUE BANDS
    # =========================================================

    def load_value_bands(
        self
    ) -> dict:

        if not self.value_bands_available():

            raise FileNotFoundError(
                "CLV value-band file not found: "
                f"{self.value_bands_path}"
            )

        with open(
            self.value_bands_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

        self.value_bands = (
            self._normalize_value_bands(
                data
            )
        )

        return self.value_bands


    # =========================================================
    # NORMALIZE VALUE BAND METADATA
    # =========================================================

    @staticmethod
    def _normalize_value_bands(
        data: dict
    ) -> dict:

        possible_low_keys = [
            "low_threshold",
            "lower_threshold",
            "q33",
            "lower_cutoff",
            "low_max"
        ]

        possible_high_keys = [
            "high_threshold",
            "upper_threshold",
            "q66",
            "upper_cutoff",
            "medium_max"
        ]

        low_threshold = None
        high_threshold = None

        # -----------------------------------------------------
        # Check top-level keys
        # -----------------------------------------------------

        for key in possible_low_keys:

            if key in data:

                low_threshold = data[
                    key
                ]

                break

        for key in possible_high_keys:

            if key in data:

                high_threshold = data[
                    key
                ]

                break

        # -----------------------------------------------------
        # Check optional nested thresholds dictionary
        # -----------------------------------------------------

        if (
            low_threshold is None
            or
            high_threshold is None
        ):

            thresholds = data.get(
                "thresholds",
                {}
            )

            if low_threshold is None:

                for key in possible_low_keys:

                    if key in thresholds:

                        low_threshold = (
                            thresholds[
                                key
                            ]
                        )

                        break

            if high_threshold is None:

                for key in possible_high_keys:

                    if key in thresholds:

                        high_threshold = (
                            thresholds[
                                key
                            ]
                        )

                        break

        if (
            low_threshold is None
            or
            high_threshold is None
        ):

            raise ValueError(
                "Unable to determine CLV "
                "value-band thresholds from "
                "metadata file."
            )

        low_threshold = float(
            low_threshold
        )

        high_threshold = float(
            high_threshold
        )

        if not np.isfinite(
            low_threshold
        ):

            raise ValueError(
                "CLV low threshold "
                "must be finite."
            )

        if not np.isfinite(
            high_threshold
        ):

            raise ValueError(
                "CLV high threshold "
                "must be finite."
            )

        if low_threshold > high_threshold:

            raise ValueError(
                "CLV lower threshold cannot "
                "be greater than upper threshold."
            )

        return {
            "low_threshold":
                low_threshold,

            "high_threshold":
                high_threshold
        }


    # =========================================================
    # ENSURE MODEL AND VALUE BANDS LOADED
    # =========================================================

    def _ensure_loaded(
        self
    ) -> None:

        if self.model is None:

            self.load_model()

        if self.value_bands is None:

            self.load_value_bands()


    # =========================================================
    # PREPARE FEATURES
    # =========================================================

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
                "Missing CLV features: "
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
        # Convert to numeric
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
        # Check invalid / missing values
        # -----------------------------------------------------

        invalid_columns = (
            X.columns[
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
        # Check infinite values
        # -----------------------------------------------------

        values = (
            X[
                self.FEATURE_COLUMNS
            ]
            .to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            values
        ).all():

            raise ValueError(
                "CLV feature values "
                "must be finite."
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
                "CLV features cannot be negative: "
                f"{negative_features}"
            )

        return X


    # =========================================================
    # PREDICT REVENUE
    # =========================================================

    def predict_revenue(
        self,
        X: pd.DataFrame
    ) -> float:

        self._ensure_loaded()

        prediction = (
            self.model
            .predict(
                X[
                    self.FEATURE_COLUMNS
                ]
            )
        )

        prediction = np.asarray(
            prediction,
            dtype=float
        ).reshape(
            -1
        )

        if len(
            prediction
        ) != 1:

            raise ValueError(
                "Unexpected CLV prediction output."
            )

        value = float(
            prediction[
                0
            ]
        )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # TransformedTargetRegressor.predict()
        # already performs inverse transformation.
        #
        # DO NOT call np.expm1() again here.
        # -----------------------------------------------------

        if self.clip_negative_predictions:

            value = max(
                value,
                0.0
            )

        return value


    # =========================================================
    # VALUE BAND
    # =========================================================

    def value_band(
        self,
        predicted_revenue: float
    ) -> str:

        self._ensure_loaded()

        low_threshold = float(
            self.value_bands[
                "low_threshold"
            ]
        )

        high_threshold = float(
            self.value_bands[
                "high_threshold"
            ]
        )

        if (
            predicted_revenue
            <=
            low_threshold
        ):

            return "Low"

        if (
            predicted_revenue
            <=
            high_threshold
        ):

            return "Medium"

        return "High"


    # =========================================================
    # COMPLETE PREDICTION
    # =========================================================

    def predict(
        self,
        features: dict
    ) -> dict:

        X = self.prepare_features(
            features
        )

        predicted_revenue = (
            self.predict_revenue(
                X
            )
        )

        clv_value_band = (
            self.value_band(
                predicted_revenue
            )
        )

        output_features = {
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

            "predicted_90_day_revenue":
                predicted_revenue,

            "clv_value_band":
                clv_value_band,

            "lower_value_threshold":
                float(
                    self.value_bands[
                        "low_threshold"
                    ]
                ),

            "upper_value_threshold":
                float(
                    self.value_bands[
                        "high_threshold"
                    ]
                ),

            "features":
                output_features
        }


    # =========================================================
    # MODEL STATUS
    # =========================================================

    def model_status(
        self
    ) -> dict:

        return {

            "model_available":
                self.model_available(),

            "model_loaded":
                self.model is not None,

            "value_bands_available":
                self.value_bands_available(),

            "value_bands_loaded":
                self.value_bands is not None,

            "model_path":
                self.model_path,

            "value_bands_path":
                self.value_bands_path,

            "feature_count":
                len(
                    self.FEATURE_COLUMNS
                ),

            "features":
                self.FEATURE_COLUMNS
        }