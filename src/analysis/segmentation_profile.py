import sys

import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class SegmentationProfiler:

    def create_cluster_profile(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Starting customer segmentation profiling."
            )

            required_columns = [
                "CustomerID",
                "Recency",
                "Frequency",
                "Monetary",
                "TotalItems",
                "AverageOrderValue",
                "Tenure",
                "Cluster"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing required columns: "
                    f"{missing_columns}"
                )

            if df.empty:

                raise ValueError(
                    "Clustered customer dataset is empty."
                )

            # ---------------------------------
            # Customer count by cluster
            # ---------------------------------

            customer_count = (
                df.groupby("Cluster")[
                    "CustomerID"
                ]
                .nunique()
            )

            # ---------------------------------
            # Customer percentage by cluster
            # ---------------------------------

            total_customers = (
                df["CustomerID"]
                .nunique()
            )

            customer_percentage = (
                customer_count
                /
                total_customers
                *
                100
            )

            # ---------------------------------
            # Features used for profiling
            # ---------------------------------

            features = [
                "Recency",
                "Frequency",
                "Monetary",
                "TotalItems",
                "AverageOrderValue",
                "Tenure"
            ]

            # ---------------------------------
            # Mean feature values
            # ---------------------------------

            mean_profile = (
                df.groupby(
                    "Cluster"
                )[features]
                .mean()
            )

            mean_profile = (
                mean_profile
                .add_suffix(
                    "_Mean"
                )
            )

            # ---------------------------------
            # Median feature values
            # ---------------------------------

            median_profile = (
                df.groupby(
                    "Cluster"
                )[features]
                .median()
            )

            median_profile = (
                median_profile
                .add_suffix(
                    "_Median"
                )
            )

            # ---------------------------------
            # Combine reports
            # ---------------------------------

            profile = pd.concat(
                [
                    customer_count.rename(
                        "CustomerCount"
                    ),
                    customer_percentage.rename(
                        "CustomerPercentage"
                    ),
                    mean_profile,
                    median_profile
                ],
                axis=1
            )

            profile = (
                profile
                .reset_index()
            )

            # ---------------------------------
            # Round numeric values
            # ---------------------------------

            numeric_columns = (
                profile
                .select_dtypes(
                    include="number"
                )
                .columns
            )

            profile[
                numeric_columns
            ] = (
                profile[
                    numeric_columns
                ]
                .round(2)
            )

            logger.info(
                "Customer segmentation profiling completed."
            )

            logger.info(
                f"Cluster profile:\n"
                f"{profile}"
            )

            return profile

        except Exception as e:

            logger.error(
                "Customer segmentation profiling failed."
            )

            raise CustomException(
                e,
                sys
            )