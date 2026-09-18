import sys

import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class SegmentationFeatureEngineering:

    def create_segmentation_features(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Starting customer segmentation feature engineering."
            )

            data = df.copy()

            # ---------------------------------
            # Standardize column names
            # ---------------------------------

            column_mapping = {
                "Customer ID": "CustomerID"
            }

            data = data.rename(
                columns=column_mapping
            )

            logger.info(
                f"Available columns after standardization: "
                f"{data.columns.tolist()}"
            )

            # ---------------------------------
            # Required columns
            # ---------------------------------

            required_columns = [
                "CustomerID",
                "Recency",
                "Frequency",
                "Monetary",
                "TotalItems",
                "AverageOrderValue",
                "Tenure"
            ]

            # ---------------------------------
            # Validate required columns
            # ---------------------------------

            missing_columns = [
                column
                for column in required_columns
                if column not in data.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing required segmentation columns: "
                    f"{missing_columns}"
                )

            # ---------------------------------
            # Select segmentation columns
            # ---------------------------------

            segmentation_df = data[
                required_columns
            ].copy()

            # ---------------------------------
            # Remove missing Customer IDs
            # ---------------------------------

            segmentation_df = (
                segmentation_df
                .dropna(
                    subset=[
                        "CustomerID"
                    ]
                )
            )

            # ---------------------------------
            # Remove rows containing
            # missing segmentation features
            # ---------------------------------

            feature_columns = [
                "Recency",
                "Frequency",
                "Monetary",
                "TotalItems",
                "AverageOrderValue",
                "Tenure"
            ]

            segmentation_df = (
                segmentation_df
                .dropna(
                    subset=feature_columns
                )
            )

            # ---------------------------------
            # Remove duplicate customers
            # ---------------------------------

            segmentation_df = (
                segmentation_df
                .drop_duplicates(
                    subset=[
                        "CustomerID"
                    ],
                    keep="first"
                )
            )

            # ---------------------------------
            # Validate numeric feature columns
            # ---------------------------------

            for feature in feature_columns:

                if not pd.api.types.is_numeric_dtype(
                    segmentation_df[feature]
                ):

                    segmentation_df[
                        feature
                    ] = pd.to_numeric(
                        segmentation_df[
                            feature
                        ],
                        errors="coerce"
                    )

            # ---------------------------------
            # Check conversion-created NaN
            # ---------------------------------

            invalid_numeric_values = (
                segmentation_df[
                    feature_columns
                ]
                .isnull()
                .sum()
            )

            if (
                invalid_numeric_values.sum()
                > 0
            ):

                raise ValueError(
                    "Invalid numeric values detected "
                    f"in segmentation features:\n"
                    f"{invalid_numeric_values}"
                )

            # ---------------------------------
            # Validate negative values
            # ---------------------------------

            negative_values = {}

            for feature in feature_columns:

                negative_count = int(
                    (
                        segmentation_df[
                            feature
                        ] < 0
                    ).sum()
                )

                negative_values[
                    feature
                ] = negative_count

            invalid_negative_features = {
                feature: count
                for feature, count
                in negative_values.items()
                if count > 0
            }

            if invalid_negative_features:

                raise ValueError(
                    "Negative values detected in "
                    "segmentation features: "
                    f"{invalid_negative_features}"
                )

            # ---------------------------------
            # Reset index
            # ---------------------------------

            segmentation_df = (
                segmentation_df
                .reset_index(
                    drop=True
                )
            )

            # ---------------------------------
            # Final validation
            # ---------------------------------

            if (
                segmentation_df[
                    "CustomerID"
                ]
                .duplicated()
                .any()
            ):

                raise ValueError(
                    "Duplicate CustomerID values "
                    "remain after preprocessing."
                )

            if segmentation_df.empty:

                raise ValueError(
                    "Segmentation dataset is empty "
                    "after preprocessing."
                )

            logger.info(
                "Customer segmentation feature "
                "engineering completed."
            )

            logger.info(
                f"Segmentation dataset shape: "
                f"{segmentation_df.shape}"
            )

            logger.info(
                f"Segmentation columns: "
                f"{segmentation_df.columns.tolist()}"
            )

            return segmentation_df

        except Exception as e:

            logger.error(
                "Customer segmentation feature "
                "engineering failed."
            )

            raise CustomException(
                e,
                sys
            )