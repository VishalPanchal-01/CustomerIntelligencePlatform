import sys

import numpy as np
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class CLVAnalysis:

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

    def validate_dataset(
        self,
        df: pd.DataFrame
    ) -> None:

        try:

            required_columns = (
                ["Customer ID"]
                +
                self.features
                +
                [self.target]
            )

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing required CLV columns: "
                    f"{missing_columns}"
                )

            if df.empty:

                raise ValueError(
                    "CLV dataset is empty."
                )

            logger.info(
                "CLV dataset validation completed."
            )

        except Exception as e:

            logger.error(
                "CLV dataset validation failed."
            )

            raise CustomException(
                e,
                sys
            )

    def analyze_target_statistics(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Analyzing FutureRevenue statistics."
            )

            self.validate_dataset(
                df
            )

            target = (
                df[self.target]
            )

            statistics = {
                "Count":
                    int(
                        target.count()
                    ),

                "Mean":
                    float(
                        target.mean()
                    ),

                "Median":
                    float(
                        target.median()
                    ),

                "StandardDeviation":
                    float(
                        target.std()
                    ),

                "Minimum":
                    float(
                        target.min()
                    ),

                "Maximum":
                    float(
                        target.max()
                    ),

                "Skewness":
                    float(
                        target.skew()
                    )
            }

            result = pd.DataFrame(
                [statistics]
            )

            logger.info(
                f"FutureRevenue statistics:\n"
                f"{result}"
            )

            return result

        except Exception as e:

            logger.error(
                "FutureRevenue statistics analysis failed."
            )

            raise CustomException(
                e,
                sys
            )

    def analyze_zero_revenue(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Analyzing zero future revenue customers."
            )

            self.validate_dataset(
                df
            )

            total_customers = (
                df["Customer ID"]
                .nunique()
            )

            zero_customers = int(
                (
                    df[self.target]
                    == 0
                )
                .sum()
            )

            positive_customers = int(
                (
                    df[self.target]
                    > 0
                )
                .sum()
            )

            zero_percentage = (
                zero_customers
                /
                total_customers
                *
                100
            )

            positive_percentage = (
                positive_customers
                /
                total_customers
                *
                100
            )

            result = pd.DataFrame(
                {
                    "RevenueGroup": [
                        "Zero Future Revenue",
                        "Positive Future Revenue"
                    ],

                    "CustomerCount": [
                        zero_customers,
                        positive_customers
                    ],

                    "Percentage": [
                        zero_percentage,
                        positive_percentage
                    ]
                }
            )

            result[
                "Percentage"
            ] = (
                result[
                    "Percentage"
                ]
                .round(2)
            )

            logger.info(
                f"Zero revenue analysis:\n"
                f"{result}"
            )

            return result

        except Exception as e:

            logger.error(
                "Zero revenue analysis failed."
            )

            raise CustomException(
                e,
                sys
            )

    def analyze_target_quantiles(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Analyzing FutureRevenue quantiles."
            )

            self.validate_dataset(
                df
            )

            quantile_values = [
                0.00,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                1.00
            ]

            quantiles = (
                df[self.target]
                .quantile(
                    quantile_values
                )
            )

            result = pd.DataFrame(
                {
                    "Quantile":
                        [
                            "0%",
                            "25%",
                            "50%",
                            "75%",
                            "90%",
                            "95%",
                            "99%",
                            "100%"
                        ],

                    "FutureRevenue":
                        quantiles.values
                }
            )

            result[
                "FutureRevenue"
            ] = (
                result[
                    "FutureRevenue"
                ]
                .round(2)
            )

            logger.info(
                f"FutureRevenue quantiles:\n"
                f"{result}"
            )

            return result

        except Exception as e:

            logger.error(
                "FutureRevenue quantile analysis failed."
            )

            raise CustomException(
                e,
                sys
            )

    def analyze_target_outliers(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Analyzing FutureRevenue outliers."
            )

            self.validate_dataset(
                df
            )

            target = (
                df[self.target]
            )

            q1 = float(
                target.quantile(
                    0.25
                )
            )

            q3 = float(
                target.quantile(
                    0.75
                )
            )

            iqr = (
                q3 - q1
            )

            lower_bound = (
                q1
                -
                1.5 * iqr
            )

            upper_bound = (
                q3
                +
                1.5 * iqr
            )

            outlier_mask = (
                (target < lower_bound)
                |
                (target > upper_bound)
            )

            outlier_count = int(
                outlier_mask.sum()
            )

            outlier_percentage = (
                outlier_count
                /
                len(df)
                *
                100
            )

            result = pd.DataFrame(
                {
                    "Q1": [
                        q1
                    ],

                    "Q3": [
                        q3
                    ],

                    "IQR": [
                        iqr
                    ],

                    "LowerBound": [
                        lower_bound
                    ],

                    "UpperBound": [
                        upper_bound
                    ],

                    "OutlierCount": [
                        outlier_count
                    ],

                    "OutlierPercentage": [
                        outlier_percentage
                    ]
                }
            )

            numeric_columns = (
                result
                .select_dtypes(
                    include="number"
                )
                .columns
            )

            result[
                numeric_columns
            ] = (
                result[
                    numeric_columns
                ]
                .round(2)
            )

            logger.info(
                f"FutureRevenue outlier analysis:\n"
                f"{result}"
            )

            return result

        except Exception as e:

            logger.error(
                "FutureRevenue outlier analysis failed."
            )

            raise CustomException(
                e,
                sys
            )

    def analyze_feature_statistics(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Analyzing CLV feature statistics."
            )

            self.validate_dataset(
                df
            )

            result = (
                df[
                    self.features
                ]
                .agg(
                    [
                        "mean",
                        "median",
                        "std",
                        "min",
                        "max",
                        "skew"
                    ]
                )
                .T
                .reset_index()
                .rename(
                    columns={
                        "index":
                            "Feature"
                    }
                )
            )

            numeric_columns = (
                result
                .select_dtypes(
                    include="number"
                )
                .columns
            )

            result[
                numeric_columns
            ] = (
                result[
                    numeric_columns
                ]
                .round(2)
            )

            logger.info(
                f"CLV feature statistics:\n"
                f"{result}"
            )

            return result

        except Exception as e:

            logger.error(
                "CLV feature statistics analysis failed."
            )

            raise CustomException(
                e,
                sys
            )

    def analyze_feature_correlation(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Analyzing CLV feature correlations."
            )

            self.validate_dataset(
                df
            )

            columns = (
                self.features
                +
                [self.target]
            )

            correlation_matrix = (
                df[
                    columns
                ]
                .corr()
            )

            target_correlation = (
                correlation_matrix[
                    self.target
                ]
                .drop(
                    self.target
                )
                .sort_values(
                    ascending=False
                )
                .reset_index()
            )

            target_correlation.columns = [
                "Feature",
                "CorrelationWithFutureRevenue"
            ]

            target_correlation[
                "CorrelationWithFutureRevenue"
            ] = (
                target_correlation[
                    "CorrelationWithFutureRevenue"
                ]
                .round(4)
            )

            logger.info(
                f"Feature correlation with FutureRevenue:\n"
                f"{target_correlation}"
            )

            return target_correlation

        except Exception as e:

            logger.error(
                "CLV correlation analysis failed."
            )

            raise CustomException(
                e,
                sys
            )

    def analyze_data_quality(
        self,
        df: pd.DataFrame
    ) -> dict:

        try:

            logger.info(
                "Analyzing CLV dataset quality."
            )

            self.validate_dataset(
                df
            )

            analysis_columns = (
                self.features
                +
                [self.target]
            )

            missing_values = (
                df[
                    analysis_columns
                ]
                .isnull()
                .sum()
                .to_dict()
            )

            numeric_data = (
                df[
                    analysis_columns
                ]
                .to_numpy(
                    dtype=float
                )
            )

            infinite_count = int(
                (
                    ~np.isfinite(
                        numeric_data
                    )
                )
                .sum()
            )

            duplicate_customers = int(
                df[
                    "Customer ID"
                ]
                .duplicated()
                .sum()
            )

            negative_target_count = int(
                (
                    df[self.target]
                    < 0
                )
                .sum()
            )

            result = {
                "missing_values":
                    {
                        key:
                            int(value)
                        for key, value
                        in missing_values.items()
                    },

                "infinite_values":
                    infinite_count,

                "duplicate_customers":
                    duplicate_customers,

                "negative_future_revenue":
                    negative_target_count
            }

            logger.info(
                f"CLV data quality analysis: "
                f"{result}"
            )

            return result

        except Exception as e:

            logger.error(
                "CLV data quality analysis failed."
            )

            raise CustomException(
                e,
                sys
            )