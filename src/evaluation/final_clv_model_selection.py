import sys

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.utils.exception import CustomException
from src.utils.logger import logger


class FinalCLVModelSelector:

    def select_best_candidate(
        self,
        candidates: dict
    ) -> dict:

        try:

            logger.info(
                "Starting final CLV candidate selection."
            )

            if not candidates:

                raise ValueError(
                    "No CLV candidates were provided."
                )

            candidate_results = []

            for candidate_name, candidate_info in candidates.items():

                search = candidate_info[
                    "search"
                ]

                model_name = candidate_info[
                    "model_name"
                ]

                target_type = candidate_info[
                    "target_type"
                ]

                # RandomizedSearchCV reports
                # negative MAE because larger
                # sklearn scores are better.
                cv_mae = float(
                    -search.best_score_
                )

                candidate_results.append(
                    {
                        "candidate_name":
                            candidate_name,

                        "model_name":
                            model_name,

                        "target_type":
                            target_type,

                        "cv_mae":
                            cv_mae,

                        "best_estimator":
                            search.best_estimator_,

                        "best_params":
                            search.best_params_
                    }
                )

            # ---------------------------------
            # Select using CV only
            # ---------------------------------

            selected = min(
                candidate_results,
                key=lambda item:
                    item["cv_mae"]
            )

            logger.info(
                f"Selected CLV candidate: "
                f"{selected['candidate_name']}"
            )

            logger.info(
                f"Selected candidate CV MAE: "
                f"{selected['cv_mae']:.4f}"
            )

            return selected

        except Exception as e:

            logger.error(
                "Final CLV candidate selection failed."
            )

            raise CustomException(
                e,
                sys
            )

    def evaluate_final_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> dict:

        try:

            logger.info(
                "Starting final CLV test evaluation."
            )

            predictions = model.predict(
                X_test
            )

            predictions = np.asarray(
                predictions,
                dtype=float
            )

            actual = np.asarray(
                y_test,
                dtype=float
            )

            if len(predictions) != len(actual):

                raise ValueError(
                    "Prediction and actual target "
                    "lengths do not match."
                )

            if not np.isfinite(
                predictions
            ).all():

                raise ValueError(
                    "Final CLV predictions contain "
                    "invalid or infinite values."
                )

            # ---------------------------------
            # Count unrealistic negatives
            # before clipping
            # ---------------------------------

            negative_count = int(
                (
                    predictions < 0
                )
                .sum()
            )

            negative_percentage = float(
                (
                    negative_count
                    /
                    len(predictions)
                )
                * 100
            )

            # ---------------------------------
            # Business-safe predictions
            # ---------------------------------

            safe_predictions = np.maximum(
                predictions,
                0
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

            result = {
                "mae":
                    mae,

                "rmse":
                    rmse,

                "r2":
                    r2,

                "negative_prediction_count":
                    negative_count,

                "negative_prediction_percentage":
                    negative_percentage,

                "actual_mean":
                    float(
                        np.mean(
                            actual
                        )
                    ),

                "predicted_mean":
                    float(
                        np.mean(
                            safe_predictions
                        )
                    ),

                "actual_median":
                    float(
                        np.median(
                            actual
                        )
                    ),

                "predicted_median":
                    float(
                        np.median(
                            safe_predictions
                        )
                    ),

                "predictions":
                    safe_predictions
            }

            logger.info(
                "Final CLV model evaluation completed."
            )

            logger.info(
                f"MAE: {mae:.4f}"
            )

            logger.info(
                f"RMSE: {rmse:.4f}"
            )

            logger.info(
                f"R2: {r2:.4f}"
            )

            return result

        except Exception as e:

            logger.error(
                "Final CLV model evaluation failed."
            )

            raise CustomException(
                e,
                sys
            )