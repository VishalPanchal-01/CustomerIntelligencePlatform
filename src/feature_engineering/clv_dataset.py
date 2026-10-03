import sys

import pandas as pd

from src.feature_engineering.customer_features import (
    CustomerFeatureEngineering
)

from src.utils.exception import CustomException
from src.utils.logger import logger


class CLVDatasetBuilder:

    def __init__(
        self,
        prediction_days: int = 90
    ):

        self.prediction_days = (
            prediction_days
        )

    def build_dataset(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Starting CLV dataset creation."
            )

            data = df.copy()

            # ---------------------------------
            # Validate required columns
            # ---------------------------------

            required_columns = [
                "Customer ID",
                "Invoice",
                "InvoiceDate",
                "Quantity",
                "Revenue"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in data.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing required columns for CLV: "
                    f"{missing_columns}"
                )

            # ---------------------------------
            # Prepare InvoiceDate
            # ---------------------------------

            data[
                "InvoiceDate"
            ] = pd.to_datetime(
                data["InvoiceDate"],
                errors="coerce"
            )

            data = (
                data
                .dropna(
                    subset=[
                        "InvoiceDate",
                        "Customer ID"
                    ]
                )
            )

            # ---------------------------------
            # Validate dataset
            # ---------------------------------

            if data.empty:

                raise ValueError(
                    "CLV source dataset is empty "
                    "after validation."
                )

            # ---------------------------------
            # Dataset end date
            # ---------------------------------

            dataset_end_date = (
                data[
                    "InvoiceDate"
                ]
                .max()
            )

            # ---------------------------------
            # Observation cutoff
            # ---------------------------------

            observation_cutoff = (
                dataset_end_date
                -
                pd.Timedelta(
                    days=self.prediction_days
                )
            )

            logger.info(
                f"Dataset end date: "
                f"{dataset_end_date}"
            )

            logger.info(
                f"Observation cutoff: "
                f"{observation_cutoff}"
            )

            # ---------------------------------
            # Observation period
            # ---------------------------------

            observation_data = (
                data[
                    data[
                        "InvoiceDate"
                    ]
                    <=
                    observation_cutoff
                ]
                .copy()
            )

            # ---------------------------------
            # Prediction period
            # ---------------------------------

            prediction_data = (
                data[
                    data[
                        "InvoiceDate"
                    ]
                    >
                    observation_cutoff
                ]
                .copy()
            )

            if observation_data.empty:

                raise ValueError(
                    "Observation period contains no data."
                )

            if prediction_data.empty:

                raise ValueError(
                    "Prediction period contains no data."
                )

            # ---------------------------------
            # Build observation features
            # ---------------------------------

            feature_engineering = (
                CustomerFeatureEngineering()
            )

            customer_features = (
                feature_engineering
                .create_customer_features(
                    observation_data,
                    reference_date=
                        observation_cutoff
                )
            )

            # ---------------------------------
            # Future revenue target
            # ---------------------------------

            future_revenue = (
                prediction_data
                .groupby(
                    "Customer ID"
                )[
                    "Revenue"
                ]
                .sum()
                .rename(
                    "FutureRevenue"
                )
                .reset_index()
            )

            # ---------------------------------
            # Merge features and target
            # ---------------------------------

            clv_dataset = (
                customer_features
                .merge(
                    future_revenue,
                    on="Customer ID",
                    how="left"
                )
            )

            # ---------------------------------
            # Customers without future purchase
            # receive future revenue = 0
            # ---------------------------------

            clv_dataset[
                "FutureRevenue"
            ] = (
                clv_dataset[
                    "FutureRevenue"
                ]
                .fillna(0)
            )

            # ---------------------------------
            # Protect against invalid
            # negative CLV target
            # ---------------------------------

            if (
                clv_dataset[
                    "FutureRevenue"
                ]
                < 0
            ).any():

                raise ValueError(
                    "FutureRevenue contains negative values."
                )

            # ---------------------------------
            # Reset index
            # ---------------------------------

            clv_dataset = (
                clv_dataset
                .reset_index(
                    drop=True
                )
            )

            logger.info(
                "CLV dataset created successfully."
            )

            logger.info(
                f"CLV dataset shape: "
                f"{clv_dataset.shape}"
            )

            logger.info(
                f"Customers with future revenue: "
                f"{(
                    clv_dataset['FutureRevenue']
                    > 0
                ).sum()}"
            )

            logger.info(
                f"Customers with zero future revenue: "
                f"{(
                    clv_dataset['FutureRevenue']
                    == 0
                ).sum()}"
            )

            return clv_dataset

        except Exception as e:

            logger.error(
                "CLV dataset creation failed."
            )

            raise CustomException(
                e,
                sys
            )