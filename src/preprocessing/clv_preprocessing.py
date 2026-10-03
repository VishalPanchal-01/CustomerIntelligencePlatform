import sys

import numpy as np
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class CLVPreprocessor:

    def __init__(self):

        self.features = [
            "Recency",
            "Frequency",
            "Monetary",
            "TotalItems",
            "AverageOrderValue",
            "Tenure"
        ]

        self.target = "FutureRevenue"

    def prepare_features(
        self,
        df: pd.DataFrame
    ):

        try:

            logger.info(
                "Starting CLV feature preprocessing."
            )

            data = df.copy()

            # ---------------------------------
            # Validate required columns
            # ---------------------------------

            required_columns = (
                self.features
                +
                [self.target]
            )

            missing_columns = [
                column
                for column in required_columns
                if column not in data.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing required CLV columns: "
                    f"{missing_columns}"
                )

            if data.empty:

                raise ValueError(
                    "CLV dataset is empty."
                )

            # ---------------------------------
            # Select model features
            # ---------------------------------

            X = (
                data[
                    self.features
                ]
                .copy()
            )

            # ---------------------------------
            # Raw target
            # ---------------------------------

            y_raw = (
                data[
                    self.target
                ]
                .copy()
            )

            # ---------------------------------
            # Validate missing values
            # ---------------------------------

            if X.isnull().any().any():

                raise ValueError(
                    "CLV features contain missing values."
                )

            if y_raw.isnull().any():

                raise ValueError(
                    "CLV target contains missing values."
                )

            # ---------------------------------
            # Validate numeric feature values
            # ---------------------------------

            feature_array = (
                X.to_numpy(
                    dtype=float
                )
            )

            if not np.isfinite(
                feature_array
            ).all():

                raise ValueError(
                    "CLV features contain infinite values."
                )

            # ---------------------------------
            # Validate target values
            # ---------------------------------

            target_array = (
                y_raw.to_numpy(
                    dtype=float
                )
            )

            if not np.isfinite(
                target_array
            ).all():

                raise ValueError(
                    "CLV target contains infinite values."
                )

            # ---------------------------------
            # Validate negative features
            # ---------------------------------

            if (
                X < 0
            ).any().any():

                raise ValueError(
                    "CLV features cannot contain negative values."
                )

            # ---------------------------------
            # Validate negative target
            # ---------------------------------

            if (
                y_raw < 0
            ).any():

                raise ValueError(
                    "FutureRevenue cannot contain negative values."
                )

            # ---------------------------------
            # Log-transformed target
            # ---------------------------------

            y_log = (
                np.log1p(
                    y_raw
                )
            )

            y_log = pd.Series(
                y_log,
                index=y_raw.index,
                name="LogFutureRevenue"
            )

            logger.info(
                "CLV preprocessing completed successfully."
            )

            logger.info(
                f"CLV feature shape: "
                f"{X.shape}"
            )

            return (
                X,
                y_raw,
                y_log
            )

        except Exception as e:

            logger.error(
                "CLV preprocessing failed."
            )

            raise CustomException(
                e,
                sys
            )