import sys

import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class UnifiedCustomerDatasetBuilder:

    CUSTOMER_ID = "Customer ID"

    # =========================================================
    # NORMALIZE CUSTOMER ID
    # =========================================================

    def normalize_customer_id(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        """
        Standardizes the customer identifier column.

        Final standard:
            Customer ID

        Older project artifacts may contain:
            CustomerID
            Customer Id

        These are automatically renamed.
        """

        try:

            data = df.copy()

            # ---------------------------------------------
            # Old project naming
            # CustomerID -> Customer ID
            # ---------------------------------------------

            if (
                "CustomerID" in data.columns
                and
                self.CUSTOMER_ID not in data.columns
            ):

                data = data.rename(
                    columns={
                        "CustomerID":
                            self.CUSTOMER_ID
                    }
                )

            # ---------------------------------------------
            # Alternative naming
            # Customer Id -> Customer ID
            # ---------------------------------------------

            if (
                "Customer Id" in data.columns
                and
                self.CUSTOMER_ID not in data.columns
            ):

                data = data.rename(
                    columns={
                        "Customer Id":
                            self.CUSTOMER_ID
                    }
                )

            # ---------------------------------------------
            # Validate
            # ---------------------------------------------

            if (
                self.CUSTOMER_ID
                not in data.columns
            ):

                raise ValueError(
                    "Customer ID column not found."
                )

            return data

        except Exception as e:

            logger.error(
                "Customer ID normalization failed."
            )

            raise CustomException(
                e,
                sys
            )

    # =========================================================
    # BUILD UNIFIED CUSTOMER DATASET
    # =========================================================

    def build(
        self,
        segmentation_df: pd.DataFrame,
        churn_df: pd.DataFrame = None,
        clv_df: pd.DataFrame = None,
        recommendation_df: pd.DataFrame = None
    ) -> pd.DataFrame:

        """
        Creates one unified customer-level dataset.

        Master customer population:
            Segmentation dataset

        Optional modules joined:
            Churn
            CLV
            Recommendations

        Final output:
            One row per Customer ID
        """

        try:

            logger.info(
                "Starting unified customer "
                "intelligence dataset creation."
            )

            # =================================================
            # SEGMENTATION
            # Master customer dataset
            # =================================================

            segmentation = (
                self.normalize_customer_id(
                    segmentation_df
                )
            )

            unified = (
                self._prepare_segmentation(
                    segmentation
                )
            )

            # =================================================
            # CHURN
            # =================================================

            if churn_df is not None:

                churn = (
                    self.normalize_customer_id(
                        churn_df
                    )
                )

                churn = (
                    self._prepare_churn(
                        churn
                    )
                )

                unified = unified.merge(
                    churn,
                    on=self.CUSTOMER_ID,
                    how="left"
                )

            # =================================================
            # CLV
            # =================================================

            if clv_df is not None:

                clv = (
                    self.normalize_customer_id(
                        clv_df
                    )
                )

                clv = (
                    self._prepare_clv(
                        clv
                    )
                )

                unified = unified.merge(
                    clv,
                    on=self.CUSTOMER_ID,
                    how="left"
                )

            # =================================================
            # RECOMMENDATIONS
            # =================================================

            if recommendation_df is not None:

                recommendations = (
                    self.normalize_customer_id(
                        recommendation_df
                    )
                )

                recommendations = (
                    self._prepare_recommendations(
                        recommendations
                    )
                )

                unified = unified.merge(
                    recommendations,
                    on=self.CUSTOMER_ID,
                    how="left"
                )

            # =================================================
            # DUPLICATE CHECK
            # =================================================

            duplicate_count = (
                unified[
                    self.CUSTOMER_ID
                ]
                .duplicated()
                .sum()
            )

            if duplicate_count > 0:

                raise ValueError(
                    f"Duplicate Customer ID records "
                    f"detected: {duplicate_count}"
                )

            # =================================================
            # SORT
            # =================================================

            unified = (
                unified
                .sort_values(
                    by=self.CUSTOMER_ID
                )
                .reset_index(
                    drop=True
                )
            )

            logger.info(
                f"Unified customer dataset created. "
                f"Shape: {unified.shape}"
            )

            return unified

        except Exception as e:

            logger.error(
                "Unified customer dataset "
                "creation failed."
            )

            raise CustomException(
                e,
                sys
            )

    # =========================================================
    # PREPARE SEGMENTATION DATA
    # =========================================================

    def _prepare_segmentation(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        """
        Prepares customer segmentation output.

        Keeps:
            Customer ID
            Customer Segment
            RFM/customer features when available
        """

        data = df.copy()

        # ---------------------------------------------
        # Detect segment column
        # ---------------------------------------------

        segment_column = (
            self._find_column(
                data,
                [
                    "Segment",
                    "SegmentName",
                    "Segment Name",
                    "CustomerSegment",
                    "Customer Segment"
                ]
            )
        )

        # ---------------------------------------------
        # Start with Customer ID
        # ---------------------------------------------

        result = data[
            [
                self.CUSTOMER_ID
            ]
        ].copy()

        # ---------------------------------------------
        # Segment
        # ---------------------------------------------

        if segment_column is not None:

            result[
                "Customer Segment"
            ] = data[
                segment_column
            ]

        # ---------------------------------------------
        # Customer behavior features
        # ---------------------------------------------

        optional_features = [
            "Recency",
            "Frequency",
            "Monetary",
            "TotalItems",
            "AverageOrderValue",
            "Tenure"
        ]

        for column in optional_features:

            if column in data.columns:

                result[
                    column
                ] = data[
                    column
                ]

        # ---------------------------------------------
        # One row per customer
        # ---------------------------------------------

        result = (
            result
            .drop_duplicates(
                subset=[
                    self.CUSTOMER_ID
                ]
            )
            .reset_index(
                drop=True
            )
        )

        return result

    # =========================================================
    # PREPARE CHURN DATA
    # =========================================================

    def _prepare_churn(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        """
        Prepares customer churn predictions.

        Supports different historical column names.
        """

        result = df[
            [
                self.CUSTOMER_ID
            ]
        ].copy()

        # ---------------------------------------------
        # Churn Probability
        # ---------------------------------------------

        probability_column = (
            self._find_column(
                df,
                [
                    "ChurnProbability",
                    "Churn Probability",
                    "Churn_Probability",
                    "Probability"
                ]
            )
        )

        # ---------------------------------------------
        # Churn Prediction
        # ---------------------------------------------

        prediction_column = (
            self._find_column(
                df,
                [
                    "ChurnPrediction",
                    "Churn Prediction",
                    "PredictedChurn",
                    "Predicted Churn",
                    "Prediction",
                    "Churn"
                ]
            )
        )

        # ---------------------------------------------
        # Churn Risk
        # ---------------------------------------------

        risk_column = (
            self._find_column(
                df,
                [
                    "ChurnRisk",
                    "Churn Risk",
                    "RiskLevel",
                    "Risk Level"
                ]
            )
        )

        if probability_column is not None:

            result[
                "Churn Probability"
            ] = df[
                probability_column
            ]

        if prediction_column is not None:

            result[
                "Churn Prediction"
            ] = df[
                prediction_column
            ]

        if risk_column is not None:

            result[
                "Churn Risk"
            ] = df[
                risk_column
            ]

        result = (
            result
            .drop_duplicates(
                subset=[
                    self.CUSTOMER_ID
                ]
            )
            .reset_index(
                drop=True
            )
        )

        return result

    # =========================================================
    # PREPARE CLV DATA
    # =========================================================

    def _prepare_clv(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        """
        Prepares CLV prediction output.
        """

        result = df[
            [
                self.CUSTOMER_ID
            ]
        ].copy()

        # ---------------------------------------------
        # Revenue prediction
        # ---------------------------------------------

        revenue_column = (
            self._find_column(
                df,
                [
                    "PredictedFutureRevenue",
                    "Predicted Future Revenue",
                    "PredictedCLV",
                    "Predicted CLV",
                    "Predicted 90-Day Revenue"
                ]
            )
        )

        # ---------------------------------------------
        # CLV Value Band
        # ---------------------------------------------

        value_band_column = (
            self._find_column(
                df,
                [
                    "CLVValueBand",
                    "CLV Value Band",
                    "ValueBand",
                    "Value Band"
                ]
            )
        )

        # ---------------------------------------------
        # Prediction Horizon
        # ---------------------------------------------

        horizon_column = (
            self._find_column(
                df,
                [
                    "PredictionHorizonDays",
                    "Prediction Horizon Days",
                    "CLV Prediction Horizon Days"
                ]
            )
        )

        if revenue_column is not None:

            result[
                "Predicted 90-Day Revenue"
            ] = df[
                revenue_column
            ]

        if value_band_column is not None:

            result[
                "CLV Value Band"
            ] = df[
                value_band_column
            ]

        if horizon_column is not None:

            result[
                "CLV Prediction Horizon Days"
            ] = df[
                horizon_column
            ]

        result = (
            result
            .drop_duplicates(
                subset=[
                    self.CUSTOMER_ID
                ]
            )
            .reset_index(
                drop=True
            )
        )

        return result

    # =========================================================
    # PREPARE RECOMMENDATIONS
    # =========================================================

    def _prepare_recommendations(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        """
        Converts multiple recommendation rows per customer
        into one customer-level recommendation record.

        Example input:

            Customer ID | StockCode | Rank
            101         | P1        | 1
            101         | P2        | 2

        Output:

            Customer ID
            Top Recommended Product
            Recommended Products
            ...
        """

        data = df.copy()

        # ---------------------------------------------
        # Detect columns
        # ---------------------------------------------

        stock_column = (
            self._find_column(
                data,
                [
                    "StockCode",
                    "Stock Code"
                ]
            )
        )

        description_column = (
            self._find_column(
                data,
                [
                    "Description",
                    "Product Description"
                ]
            )
        )

        score_column = (
            self._find_column(
                data,
                [
                    "Score",
                    "RecommendationScore",
                    "Recommendation Score",
                    "HybridScore",
                    "Hybrid Score",
                    "PopularityScore",
                    "Popularity Score"
                ]
            )
        )

        rank_column = (
            self._find_column(
                data,
                [
                    "Rank",
                    "RecommendationRank",
                    "Recommendation Rank",
                    "PopularityRank",
                    "Popularity Rank"
                ]
            )
        )

        source_column = (
            self._find_column(
                data,
                [
                    "RecommendationSource",
                    "Recommendation Source"
                ]
            )
        )

        # ---------------------------------------------
        # StockCode is required
        # ---------------------------------------------

        if stock_column is None:

            raise ValueError(
                "Recommendation product identifier "
                "column was not found."
            )

        # ---------------------------------------------
        # Standardize product ID
        # ---------------------------------------------

        data[
            stock_column
        ] = (
            data[
                stock_column
            ]
            .astype(str)
        )

        # ---------------------------------------------
        # Sort by recommendation rank
        # ---------------------------------------------

        if rank_column is not None:

            data = data.sort_values(
                by=[
                    self.CUSTOMER_ID,
                    rank_column
                ],
                ascending=[
                    True,
                    True
                ]
            )

        # =================================================
        # TOP RECOMMENDATION
        # =================================================

        first_recommendation = (
            data
            .groupby(
                self.CUSTOMER_ID,
                as_index=False
            )
            .first()
        )

        result = (
            first_recommendation[
                [
                    self.CUSTOMER_ID
                ]
            ]
            .copy()
        )

        result[
            "Top Recommended Stock Code"
        ] = (
            first_recommendation[
                stock_column
            ]
            .astype(str)
        )

        # ---------------------------------------------
        # Top product description
        # ---------------------------------------------

        if description_column is not None:

            result[
                "Top Recommended Product"
            ] = (
                first_recommendation[
                    description_column
                ]
            )

        # ---------------------------------------------
        # Top recommendation score
        # ---------------------------------------------

        if score_column is not None:

            result[
                "Top Recommendation Score"
            ] = (
                first_recommendation[
                    score_column
                ]
            )

        # ---------------------------------------------
        # Recommendation source
        # ---------------------------------------------

        if source_column is not None:

            result[
                "Recommendation Source"
            ] = (
                first_recommendation[
                    source_column
                ]
            )

        # =================================================
        # ALL RECOMMENDED PRODUCT CODES
        # =================================================

        product_lists = (
            data
            .groupby(
                self.CUSTOMER_ID
            )[
                stock_column
            ]
            .apply(
                lambda values:
                    " | ".join(
                        values
                        .astype(str)
                        .tolist()
                    )
            )
            .reset_index(
                name=
                    "Recommended Stock Codes"
            )
        )

        result = result.merge(
            product_lists,
            on=self.CUSTOMER_ID,
            how="left"
        )

        # =================================================
        # ALL PRODUCT DESCRIPTIONS
        # =================================================

        if description_column is not None:

            description_lists = (
                data
                .groupby(
                    self.CUSTOMER_ID
                )[
                    description_column
                ]
                .apply(
                    lambda values:
                        " | ".join(
                            values
                            .fillna("")
                            .astype(str)
                            .tolist()
                        )
                )
                .reset_index(
                    name=
                        "Recommended Products"
                )
            )

            result = result.merge(
                description_lists,
                on=self.CUSTOMER_ID,
                how="left"
            )

        # ---------------------------------------------
        # Ensure one row per customer
        # ---------------------------------------------

        result = (
            result
            .drop_duplicates(
                subset=[
                    self.CUSTOMER_ID
                ]
            )
            .reset_index(
                drop=True
            )
        )

        return result

    # =========================================================
    # COVERAGE ANALYSIS
    # =========================================================

    def analyze_coverage(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        """
        Measures how many customers have information
        available from each intelligence module.
        """

        try:

            if df.empty:

                raise ValueError(
                    "Unified customer dataset is empty."
                )

            if (
                self.CUSTOMER_ID
                not in df.columns
            ):

                raise ValueError(
                    "Customer ID column not found "
                    "in unified dataset."
                )

            total_customers = (
                df[
                    self.CUSTOMER_ID
                ]
                .nunique()
            )

            fields = [

                (
                    "Segmentation",
                    "Customer Segment"
                ),

                (
                    "Churn",
                    "Churn Probability"
                ),

                (
                    "Churn",
                    "Churn Prediction"
                ),

                (
                    "Churn",
                    "Churn Risk"
                ),

                (
                    "CLV",
                    "Predicted 90-Day Revenue"
                ),

                (
                    "CLV",
                    "CLV Value Band"
                ),

                (
                    "Recommendation",
                    "Top Recommended Product"
                )
            ]

            rows = []

            for (
                module,
                column
            ) in fields:

                # -----------------------------------------
                # Column completely unavailable
                # -----------------------------------------

                if column not in df.columns:

                    rows.append(
                        {
                            "Module":
                                module,

                            "Module Field":
                                column,

                            "Total Customers":
                                int(
                                    total_customers
                                ),

                            "Available Customers":
                                0,

                            "Missing Customers":
                                int(
                                    total_customers
                                ),

                            "Coverage Percentage":
                                0.0
                        }
                    )

                    continue

                # -----------------------------------------
                # Available values
                # -----------------------------------------

                available = int(
                    df[
                        column
                    ]
                    .notna()
                    .sum()
                )

                missing = int(
                    total_customers
                    -
                    available
                )

                if total_customers > 0:

                    coverage = (
                        available
                        /
                        total_customers
                        *
                        100
                    )

                else:

                    coverage = 0.0

                rows.append(
                    {
                        "Module":
                            module,

                        "Module Field":
                            column,

                        "Total Customers":
                            int(
                                total_customers
                            ),

                        "Available Customers":
                            available,

                        "Missing Customers":
                            missing,

                        "Coverage Percentage":
                            float(
                                coverage
                            )
                    }
                )

            coverage_df = pd.DataFrame(
                rows
            )

            return coverage_df

        except Exception as e:

            logger.error(
                "Customer intelligence coverage "
                "analysis failed."
            )

            raise CustomException(
                e,
                sys
            )

    # =========================================================
    # DATA QUALITY ANALYSIS
    # =========================================================

    def analyze_quality(
        self,
        df: pd.DataFrame
    ) -> dict:

        """
        Basic quality checks for the unified dataset.
        """

        try:

            if df.empty:

                raise ValueError(
                    "Unified customer dataset is empty."
                )

            duplicate_customers = int(
                df[
                    self.CUSTOMER_ID
                ]
                .duplicated()
                .sum()
            )

            missing_customer_ids = int(
                df[
                    self.CUSTOMER_ID
                ]
                .isnull()
                .sum()
            )

            quality_report = {

                "row_count":
                    int(
                        len(
                            df
                        )
                    ),

                "column_count":
                    int(
                        len(
                            df.columns
                        )
                    ),

                "unique_customers":
                    int(
                        df[
                            self.CUSTOMER_ID
                        ]
                        .nunique()
                    ),

                "duplicate_customers":
                    duplicate_customers,

                "missing_customer_ids":
                    missing_customer_ids,

                "one_row_per_customer":
                    (
                        duplicate_customers
                        ==
                        0
                    )
            }

            return quality_report

        except Exception as e:

            logger.error(
                "Unified customer data quality "
                "analysis failed."
            )

            raise CustomException(
                e,
                sys
            )

    # =========================================================
    # HELPER
    # =========================================================

    def _find_column(
        self,
        df: pd.DataFrame,
        candidates: list
    ):

        """
        Returns the first matching column name.
        """

        for column in candidates:

            if column in df.columns:

                return column

        return None