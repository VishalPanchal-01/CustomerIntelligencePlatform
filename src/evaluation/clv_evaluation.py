import sys

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.utils.exception import CustomException
from src.utils.logger import logger


class CLVModelEvaluation:

    def evaluate(
        self,
        model,
        X_test,
        y_test_raw,
        target_type: str = "raw"
    ) -> dict:

        try:

            logger.info(
                f"Starting CLV model evaluation "
                f"for target type: {target_type}"
            )

            # ---------------------------------
            # Validate target type
            # ---------------------------------

            valid_target_types = [
                "raw",
                "log"
            ]

            if target_type not in valid_target_types:

                raise ValueError(
                    f"target_type must be one of "
                    f"{valid_target_types}"
                )

            # ---------------------------------
            # Generate predictions
            # ---------------------------------

            predictions = (
                model.predict(
                    X_test
                )
            )

            predictions = np.asarray(
                predictions,
                dtype=float
            )

            # ---------------------------------
            # Convert log predictions
            # back to revenue scale
            # ---------------------------------

            if target_type == "log":

                predictions = (
                    np.expm1(
                        predictions
                    )
                )

            # ---------------------------------
            # Preserve raw predictions
            # before business-safe clipping
            # ---------------------------------

            raw_predictions = (
                predictions.copy()
            )

            # ---------------------------------
            # Count negative predictions
            # ---------------------------------

            negative_prediction_count = int(
                (
                    raw_predictions < 0
                )
                .sum()
            )

            negative_prediction_percentage = float(
                (
                    negative_prediction_count
                    /
                    len(raw_predictions)
                )
                * 100
            )

            # ---------------------------------
            # Business-safe predictions
            # ---------------------------------

            safe_predictions = np.maximum(
                raw_predictions,
                0
            )

            # ---------------------------------
            # Prepare actual target
            # ---------------------------------

            actual = np.asarray(
                y_test_raw,
                dtype=float
            )

            # ---------------------------------
            # MAE
            # ---------------------------------

            mae = float(
                mean_absolute_error(
                    actual,
                    safe_predictions
                )
            )

            # ---------------------------------
            # RMSE
            # ---------------------------------

            mse = float(
                mean_squared_error(
                    actual,
                    safe_predictions
                )
            )

            rmse = float(
                np.sqrt(
                    mse
                )
            )

            # ---------------------------------
            # R²
            # ---------------------------------

            r2 = float(
                r2_score(
                    actual,
                    safe_predictions
                )
            )

            # ---------------------------------
            # Additional diagnostics
            # ---------------------------------

            actual_mean = float(
                np.mean(
                    actual
                )
            )

            predicted_mean = float(
                np.mean(
                    safe_predictions
                )
            )

            actual_median = float(
                np.median(
                    actual
                )
            )

            predicted_median = float(
                np.median(
                    safe_predictions
                )
            )

            result = {

                "mae":
                    mae,

                "rmse":
                    rmse,

                "r2":
                    r2,

                "negative_prediction_count":
                    negative_prediction_count,

                "negative_prediction_percentage":
                    negative_prediction_percentage,

                "actual_mean":
                    actual_mean,

                "predicted_mean":
                    predicted_mean,

                "actual_median":
                    actual_median,

                "predicted_median":
                    predicted_median,

                "predictions":
                    safe_predictions
            }

            logger.info(
                f"CLV evaluation completed. "
                f"MAE={mae:.4f}, "
                f"RMSE={rmse:.4f}, "
                f"R2={r2:.4f}"
            )

            return result

        except Exception as e:

            logger.error(
                "CLV model evaluation failed."
            )

            raise CustomException(
                e,
                sys
            )