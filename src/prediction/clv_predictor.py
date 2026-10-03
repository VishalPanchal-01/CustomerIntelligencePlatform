import json
import os
import sys

import numpy as np
import pandas as pd

from src.utils.clv_model_persistence import (
    CLVModelPersistence
)

from src.utils.exception import CustomException
from src.utils.logger import logger


class CLVPredictor:

    def __init__(
        self,
        model_path: str =
            "models/clv/clv_model.pkl",

        value_band_path: str =
            "models/clv/clv_value_bands.json"
    ):

        try:

            logger.info(
                "Initializing CLV Predictor."
            )

            self.features = [
                "Recency",
                "Frequency",
                "Monetary",
                "TotalItems",
                "AverageOrderValue",
                "Tenure"
            ]

            self.model_path = (
                model_path
            )

            self.value_band_path = (
                value_band_path
            )

            # ---------------------------------
            # Load final model only once
            # ---------------------------------

            persistence = (
                CLVModelPersistence()
            )

            self.model = (
                persistence.load_model(
                    self.model_path
                )
            )

            # ---------------------------------
            # Load value-band thresholds
            # ---------------------------------

            self.value_bands = (
                self._load_value_bands(
                    self.value_band_path
                )
            )

            logger.info(
                "CLV Predictor initialized successfully."
            )

        except Exception as e:

            logger.error(
                "CLV Predictor initialization failed."
            )

            raise CustomException(
                e,
                sys
            )

    def predict(
        self,
        customer_data: dict
    ) -> dict:

        try:

            logger.info(
                "Starting single customer CLV prediction."
            )

            customer_df = (
                self._prepare_single_customer(
                    customer_data
                )
            )

            prediction = (
                self.model.predict(
                    customer_df
                )
            )

            prediction = float(
                prediction[0]
            )

            # ---------------------------------
            # Keep record of raw model output
            # ---------------------------------

            raw_prediction = (
                prediction
            )

            # ---------------------------------
            # Revenue cannot be negative
            # ---------------------------------

            prediction = max(
                prediction,
                0.0
            )

            value_band = (
                self._assign_value_band(
                    prediction
                )
            )

            result = {

                "predicted_future_revenue":
                    round(
                        prediction,
                        2
                    ),

                "prediction_horizon_days":
                    90,

                "value_band":
                    value_band,

                "was_negative_before_clipping":
                    bool(
                        raw_prediction < 0
                    )
            }

            logger.info(
                f"CLV prediction completed: "
                f"{result}"
            )

            return result

        except Exception as e:

            logger.error(
                "Single customer CLV prediction failed."
            )

            raise CustomException(
                e,
                sys
            )

    def predict_batch(
        self,
        customers
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Starting batch CLV prediction."
            )

            # ---------------------------------
            # Convert list/dict/DataFrame
            # ---------------------------------

            if isinstance(
                customers,
                pd.DataFrame
            ):

                data = customers.copy()

            elif isinstance(
                customers,
                list
            ):

                data = pd.DataFrame(
                    customers
                )

            elif isinstance(
                customers,
                dict
            ):

                data = pd.DataFrame(
                    [customers]
                )

            else:

                raise ValueError(
                    "customers must be a dictionary, "
                    "list of dictionaries, or DataFrame."
                )

            # ---------------------------------
            # Validate batch
            # ---------------------------------

            feature_data = (
                self._validate_dataframe(
                    data
                )
            )

            # ---------------------------------
            # Predict
            # ---------------------------------

            predictions = (
                self.model.predict(
                    feature_data
                )
            )

            predictions = np.asarray(
                predictions,
                dtype=float
            )

            # ---------------------------------
            # Invalid prediction validation
            # ---------------------------------

            if not np.isfinite(
                predictions
            ).all():

                raise ValueError(
                    "Model generated invalid "
                    "CLV predictions."
                )

            negative_flags = (
                predictions < 0
            )

            safe_predictions = np.maximum(
                predictions,
                0
            )

            # ---------------------------------
            # Build output
            # ---------------------------------

            result = data.copy()

            result[
                "PredictedFutureRevenue"
            ] = np.round(
                safe_predictions,
                2
            )

            result[
                "CLVValueBand"
            ] = [
                self._assign_value_band(
                    value
                )
                for value
                in safe_predictions
            ]

            result[
                "PredictionHorizonDays"
            ] = 90

            result[
                "WasNegativeBeforeClipping"
            ] = negative_flags

            logger.info(
                f"Batch CLV prediction completed "
                f"for {len(result)} customers."
            )

            return result

        except Exception as e:

            logger.error(
                "Batch CLV prediction failed."
            )

            raise CustomException(
                e,
                sys
            )

    def _prepare_single_customer(
        self,
        customer_data: dict
    ) -> pd.DataFrame:

        if not isinstance(
            customer_data,
            dict
        ):

            raise ValueError(
                "customer_data must be a dictionary."
            )

        df = pd.DataFrame(
            [customer_data]
        )

        return (
            self._validate_dataframe(
                df
            )
        )

    def _validate_dataframe(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        if df.empty:

            raise ValueError(
                "Customer data is empty."
            )

        # ---------------------------------
        # Required features
        # ---------------------------------

        missing_features = [
            feature
            for feature
            in self.features
            if feature not in df.columns
        ]

        if missing_features:

            raise ValueError(
                f"Missing CLV features: "
                f"{missing_features}"
            )

        feature_data = (
            df[
                self.features
            ]
            .copy()
        )

        # ---------------------------------
        # Numeric conversion validation
        # ---------------------------------

        for feature in self.features:

            feature_data[
                feature
            ] = pd.to_numeric(
                feature_data[
                    feature
                ],
                errors="coerce"
            )

        # ---------------------------------
        # Missing / invalid validation
        # ---------------------------------

        if feature_data.isnull().any().any():

            raise ValueError(
                "CLV features contain missing "
                "or non-numeric values."
            )

        numeric_values = (
            feature_data.to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            numeric_values
        ).all():

            raise ValueError(
                "CLV features contain infinite values."
            )

        # ---------------------------------
        # Business validation
        # ---------------------------------

        if (
            feature_data < 0
        ).any().any():

            raise ValueError(
                "CLV features cannot contain "
                "negative values."
            )

        return (
            feature_data[
                self.features
            ]
        )

    def _load_value_bands(
        self,
        file_path: str
    ) -> dict:

        if not os.path.exists(
            file_path
        ):

            raise FileNotFoundError(
                f"CLV value-band configuration "
                f"not found: {file_path}"
            )

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            value_bands = (
                json.load(
                    file
                )
            )

        required_keys = [
            "low_upper_bound",
            "medium_upper_bound"
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in value_bands
        ]

        if missing_keys:

            raise ValueError(
                f"Missing CLV value-band configuration: "
                f"{missing_keys}"
            )

        return value_bands

    def _assign_value_band(
        self,
        predicted_revenue: float
    ) -> str:

        low_upper_bound = float(
            self.value_bands[
                "low_upper_bound"
            ]
        )

        medium_upper_bound = float(
            self.value_bands[
                "medium_upper_bound"
            ]
        )

        if (
            predicted_revenue
            <=
            low_upper_bound
        ):

            return "Low"

        if (
            predicted_revenue
            <=
            medium_upper_bound
        ):

            return "Medium"

        return "High"