import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.utils.exception import CustomException
from src.utils.logger import logger


class SegmentationPreprocessor:

    def __init__(self):

        self.features = [
            "Recency",
            "Frequency",
            "Monetary",
            "TotalItems",
            "AverageOrderValue",
            "Tenure"
        ]

        self.log_features = [
            "Frequency",
            "Monetary",
            "TotalItems",
            "AverageOrderValue"
        ]

        self.scaler = StandardScaler()

    def prepare_features(
        self,
        df: pd.DataFrame
    ):

        try:

            logger.info(
                "Starting segmentation preprocessing."
            )

            data = df.copy()

            # ---------------------------------
            # Validate required columns
            # ---------------------------------

            required_columns = (
                ["CustomerID"]
                +
                self.features
            )

            missing_columns = [
                column
                for column in required_columns
                if column not in data.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing required columns: "
                    f"{missing_columns}"
                )

            # ---------------------------------
            # Keep customer IDs separately
            # ---------------------------------

            customer_ids = (
                data["CustomerID"]
                .copy()
            )

            # ---------------------------------
            # Select clustering features
            # ---------------------------------

            X = (
                data[
                    self.features
                ]
                .copy()
            )

            # ---------------------------------
            # Check missing values
            # ---------------------------------

            if X.isnull().any().any():

                raise ValueError(
                    "Segmentation features contain missing values."
                )

            # ---------------------------------
            # Check infinite values
            # ---------------------------------

            numeric_values = (
                X.to_numpy(
                    dtype=float
                )
            )

            if not np.isfinite(
                numeric_values
            ).all():

                raise ValueError(
                    "Segmentation features contain infinite values."
                )

            # ---------------------------------
            # Check negative values
            # ---------------------------------

            if (
                X < 0
            ).any().any():

                raise ValueError(
                    "Segmentation features cannot contain negative values."
                )

            # ---------------------------------
            # Log transformation
            # ---------------------------------

            for feature in (
                self.log_features
            ):

                X[feature] = (
                    np.log1p(
                        X[feature]
                    )
                )

            logger.info(
                "Log transformation completed."
            )

            # ---------------------------------
            # Standard scaling
            # ---------------------------------

            scaled_array = (
                self.scaler
                .fit_transform(
                    X
                )
            )

            scaled_df = (
                pd.DataFrame(
                    scaled_array,
                    columns=
                        self.features,
                    index=
                        X.index
                )
            )

            logger.info(
                "Standard scaling completed."
            )

            logger.info(
                f"Scaled segmentation "
                f"feature shape: "
                f"{scaled_df.shape}"
            )

            return (
                customer_ids,
                scaled_df
            )

        except Exception as e:

            logger.error(
                "Segmentation preprocessing failed."
            )

            raise CustomException(
                e,
                sys
            )

    def transform_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Transforming new segmentation data."
            )

            data = df.copy()

            missing_columns = [
                feature
                for feature in self.features
                if feature not in data.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing required features: "
                    f"{missing_columns}"
                )

            X = (
                data[
                    self.features
                ]
                .copy()
            )

            if X.isnull().any().any():

                raise ValueError(
                    "Segmentation features contain missing values."
                )

            numeric_values = (
                X.to_numpy(
                    dtype=float
                )
            )

            if not np.isfinite(
                numeric_values
            ).all():

                raise ValueError(
                    "Segmentation features contain infinite values."
                )

            if (
                X < 0
            ).any().any():

                raise ValueError(
                    "Segmentation features cannot contain negative values."
                )

            for feature in (
                self.log_features
            ):

                X[feature] = (
                    np.log1p(
                        X[feature]
                    )
                )

            scaled_array = (
                self.scaler
                .transform(
                    X
                )
            )

            scaled_df = (
                pd.DataFrame(
                    scaled_array,
                    columns=
                        self.features,
                    index=
                        X.index
                )
            )

            logger.info(
                "New segmentation data transformed successfully."
            )

            return scaled_df

        except Exception as e:

            logger.error(
                "Segmentation feature transformation failed."
            )

            raise CustomException(
                e,
                sys
            )