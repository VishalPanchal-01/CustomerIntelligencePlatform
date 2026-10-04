import sys

import numpy as np
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class CustomerIntelligenceGenerator:

    CUSTOMER_ID = "Customer ID"

    FEATURES = [
        "Recency",
        "Frequency",
        "Monetary",
        "TotalItems",
        "AverageOrderValue",
        "Tenure"
    ]

    def __init__(
        self,
        churn_low_threshold: float = 0.30,
        churn_high_threshold: float = 0.70
    ):

        try:

            if not (
                0
                <= churn_low_threshold
                < churn_high_threshold
                <= 1
            ):

                raise ValueError(
                    "Churn thresholds must satisfy: "
                    "0 <= low < high <= 1."
                )

            self.churn_low_threshold = (
                churn_low_threshold
            )

            self.churn_high_threshold = (
                churn_high_threshold
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )

    # =========================================================
    # NORMALIZE CUSTOMER ID
    # =========================================================

    def normalize_customer_id(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        try:

            data = df.copy()

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
    # PREPARE CUSTOMER FEATURES
    # =========================================================

    def prepare_customer_features(
        self,
        customer_df: pd.DataFrame
    ) -> tuple:

        try:

            data = (
                self.normalize_customer_id(
                    customer_df
                )
            )

            if data.empty:

                raise ValueError(
                    "Customer feature dataset is empty."
                )

            missing_features = [
                feature
                for feature in self.FEATURES
                if feature not in data.columns
            ]

            if missing_features:

                raise ValueError(
                    f"Missing customer features: "
                    f"{missing_features}"
                )

            # ---------------------------------------------
            # Keep one row per customer
            # ---------------------------------------------

            data = (
                data
                .drop_duplicates(
                    subset=[
                        self.CUSTOMER_ID
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            feature_data = (
                data[
                    self.FEATURES
                ]
                .copy()
            )

            # ---------------------------------------------
            # Convert to numeric
            # ---------------------------------------------

            for feature in self.FEATURES:

                feature_data[
                    feature
                ] = pd.to_numeric(
                    feature_data[
                        feature
                    ],
                    errors="coerce"
                )

            # ---------------------------------------------
            # Missing values
            # ---------------------------------------------

            if (
                feature_data
                .isnull()
                .any()
                .any()
            ):

                missing = (
                    feature_data
                    .isnull()
                    .sum()
                )

                missing = (
                    missing[
                        missing > 0
                    ]
                    .to_dict()
                )

                raise ValueError(
                    f"Customer features contain "
                    f"missing values: {missing}"
                )

            # ---------------------------------------------
            # Infinite values
            # ---------------------------------------------

            values = (
                feature_data
                .to_numpy(
                    dtype=float
                )
            )

            if not np.isfinite(
                values
            ).all():

                raise ValueError(
                    "Customer features contain "
                    "infinite values."
                )

            # ---------------------------------------------
            # Negative values
            # ---------------------------------------------

            if (
                feature_data < 0
            ).any().any():

                raise ValueError(
                    "Customer intelligence features "
                    "cannot contain negative values."
                )

            customer_ids = (
                data[
                    self.CUSTOMER_ID
                ]
                .copy()
            )

            return (
                data,
                feature_data,
                customer_ids
            )

        except Exception as e:

            logger.error(
                "Customer feature preparation failed."
            )

            raise CustomException(
                e,
                sys
            )

    # =========================================================
    # CHURN PREDICTIONS
    # =========================================================

    def generate_churn_predictions(
        self,
        customer_df: pd.DataFrame,
        churn_model
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Generating churn predictions "
                "for all customers."
            )

            (
                data,
                X,
                customer_ids
            ) = (
                self.prepare_customer_features(
                    customer_df
                )
            )

            # ---------------------------------------------
            # Class prediction
            # ---------------------------------------------

            predictions = (
                churn_model.predict(
                    X
                )
            )

            predictions = np.asarray(
                predictions
            ).ravel()

            # ---------------------------------------------
            # Probability prediction
            # ---------------------------------------------

            probabilities = (
                self._predict_churn_probability(
                    churn_model,
                    X
                )
            )

            if (
                len(predictions)
                !=
                len(data)
            ):

                raise ValueError(
                    "Churn prediction length "
                    "does not match customers."
                )

            if (
                len(probabilities)
                !=
                len(data)
            ):

                raise ValueError(
                    "Churn probability length "
                    "does not match customers."
                )

            # ---------------------------------------------
            # Risk bands
            # ---------------------------------------------

            risk_levels = [
                self._assign_churn_risk(
                    probability
                )
                for probability
                in probabilities
            ]

            result = pd.DataFrame(
                {
                    self.CUSTOMER_ID:
                        customer_ids.values,

                    "Churn Probability":
                        np.round(
                            probabilities,
                            6
                        ),

                    "Churn Prediction":
                        predictions.astype(int),

                    "Churn Risk":
                        risk_levels
                }
            )

            logger.info(
                f"Churn predictions generated "
                f"for {len(result)} customers."
            )

            return result

        except Exception as e:

            logger.error(
                "Full customer churn prediction failed."
            )

            raise CustomException(
                e,
                sys
            )

    # =========================================================
    # CHURN PROBABILITY
    # =========================================================

    def _predict_churn_probability(
        self,
        model,
        X: pd.DataFrame
    ) -> np.ndarray:

        if not hasattr(
            model,
            "predict_proba"
        ):

            raise ValueError(
                "Saved churn model does not support "
                "predict_proba()."
            )

        probabilities = (
            model.predict_proba(
                X
            )
        )

        probabilities = np.asarray(
            probabilities,
            dtype=float
        )

        if probabilities.ndim != 2:

            raise ValueError(
                "Unexpected churn probability output."
            )

        # ---------------------------------------------
        # Find class 1 probability
        # ---------------------------------------------

        classes = getattr(
            model,
            "classes_",
            None
        )

        if classes is not None:

            classes = np.asarray(
                classes
            )

            matching = np.where(
                classes == 1
            )[0]

            if len(matching) == 0:

                raise ValueError(
                    "Churn class 1 not found "
                    "in model classes."
                )

            positive_index = int(
                matching[0]
            )

        else:

            if probabilities.shape[1] < 2:

                raise ValueError(
                    "Unable to determine churn "
                    "positive-class probability."
                )

            positive_index = 1

        positive_probability = (
            probabilities[
                :,
                positive_index
            ]
        )

        if not np.isfinite(
            positive_probability
        ).all():

            raise ValueError(
                "Invalid churn probabilities generated."
            )

        return positive_probability

    # =========================================================
    # CHURN RISK BAND
    # =========================================================

    def _assign_churn_risk(
        self,
        probability: float
    ) -> str:

        probability = float(
            probability
        )

        if (
            probability
            < self.churn_low_threshold
        ):

            return "Low"

        if (
            probability
            < self.churn_high_threshold
        ):

            return "Medium"

        return "High"

    # =========================================================
    # CLV PREDICTIONS
    # =========================================================

    def generate_clv_predictions(
        self,
        customer_df: pd.DataFrame,
        clv_predictor
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Generating CLV predictions "
                "for all customers."
            )

            (
                data,
                X,
                customer_ids
            ) = (
                self.prepare_customer_features(
                    customer_df
                )
            )

            # ---------------------------------------------
            # Build predictor input
            # ---------------------------------------------

            predictor_input = (
                X.copy()
            )

            predictor_input.insert(
                0,
                self.CUSTOMER_ID,
                customer_ids.values
            )

            # ---------------------------------------------
            # Existing CLV predictor
            # ---------------------------------------------

            result = (
                clv_predictor.predict_batch(
                    predictor_input
                )
            )

            # ---------------------------------------------
            # Normalize old output if required
            # ---------------------------------------------

            result = (
                self.normalize_customer_id(
                    result
                )
            )

            # ---------------------------------------------
            # Rename production fields
            # ---------------------------------------------

            rename_mapping = {

                "PredictedFutureRevenue":
                    "Predicted 90-Day Revenue",

                "CLVValueBand":
                    "CLV Value Band",

                "PredictionHorizonDays":
                    "CLV Prediction Horizon Days",

                "WasNegativeBeforeClipping":
                    (
                        "CLV Was Negative "
                        "Before Clipping"
                    )
            }

            result = result.rename(
                columns={
                    old:
                        new
                    for old, new
                    in rename_mapping.items()
                    if old in result.columns
                }
            )

            # ---------------------------------------------
            # Output columns
            # ---------------------------------------------

            desired_columns = [
                self.CUSTOMER_ID,
                "Predicted 90-Day Revenue",
                "CLV Value Band",
                "CLV Prediction Horizon Days",
                "CLV Was Negative Before Clipping"
            ]

            available_columns = [
                column
                for column in desired_columns
                if column in result.columns
            ]

            result = result[
                available_columns
            ].copy()

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

            logger.info(
                f"CLV predictions generated "
                f"for {len(result)} customers."
            )

            return result

        except Exception as e:

            logger.error(
                "Full customer CLV prediction failed."
            )

            raise CustomException(
                e,
                sys
            )

    # =========================================================
    # RECOMMENDATIONS FOR ALL CUSTOMERS
    # =========================================================

    def generate_recommendations(
        self,
        customer_df: pd.DataFrame,
        recommendation_predictor,
        top_k: int = 5,
        mode: str = "next_purchase"
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Generating recommendations "
                "for all customers."
            )

            data = (
                self.normalize_customer_id(
                    customer_df
                )
            )

            if top_k < 1:

                raise ValueError(
                    "top_k must be at least 1."
                )

            customer_ids = (
                data[
                    self.CUSTOMER_ID
                ]
                .dropna()
                .drop_duplicates()
                .tolist()
            )

            if len(
                customer_ids
            ) == 0:

                raise ValueError(
                    "No Customer ID values available "
                    "for recommendation generation."
                )

            result = (
                recommendation_predictor
                .recommend_batch(
                    customer_ids=
                        customer_ids,

                    top_k=
                        top_k,

                    mode=
                        mode
                )
            )

            # ---------------------------------------------
            # Existing predictor output may still
            # contain CustomerID.
            # ---------------------------------------------

            result = (
                self.normalize_customer_id(
                    result
                )
            )

            # ---------------------------------------------
            # Standardize output field names
            # ---------------------------------------------

            rename_mapping = {

                "KnownCustomer":
                    "Known Customer",

                "RecommendationSource":
                    "Recommendation Source",

                "StockCode":
                    "Stock Code",

                "Score":
                    "Recommendation Score",

                "Rank":
                    "Recommendation Rank"
            }

            result = result.rename(
                columns={
                    old:
                        new
                    for old, new
                    in rename_mapping.items()
                    if old in result.columns
                }
            )

            # ---------------------------------------------
            # Validation
            # ---------------------------------------------

            required_columns = [
                self.CUSTOMER_ID,
                "Stock Code",
                "Recommendation Rank"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in result.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing recommendation output "
                    f"columns: {missing_columns}"
                )

            result = (
                result
                .sort_values(
                    by=[
                        self.CUSTOMER_ID,
                        "Recommendation Rank"
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            logger.info(
                f"Recommendation rows generated: "
                f"{len(result)}"
            )

            logger.info(
                f"Customers receiving recommendations: "
                f"{result[self.CUSTOMER_ID].nunique()}"
            )

            return result

        except Exception as e:

            logger.error(
                "Full customer recommendation "
                "generation failed."
            )

            raise CustomException(
                e,
                sys
            )

    # =========================================================
    # MODULE COVERAGE SUMMARY
    # =========================================================

    def create_generation_summary(
        self,
        customer_df: pd.DataFrame,
        churn_df: pd.DataFrame,
        clv_df: pd.DataFrame,
        recommendation_df: pd.DataFrame
    ) -> dict:

        try:

            customers = (
                self.normalize_customer_id(
                    customer_df
                )
            )

            churn = (
                self.normalize_customer_id(
                    churn_df
                )
            )

            clv = (
                self.normalize_customer_id(
                    clv_df
                )
            )

            recommendations = (
                self.normalize_customer_id(
                    recommendation_df
                )
            )

            total_customers = int(
                customers[
                    self.CUSTOMER_ID
                ]
                .nunique()
            )

            churn_customers = int(
                churn[
                    self.CUSTOMER_ID
                ]
                .nunique()
            )

            clv_customers = int(
                clv[
                    self.CUSTOMER_ID
                ]
                .nunique()
            )

            recommendation_customers = int(
                recommendations[
                    self.CUSTOMER_ID
                ]
                .nunique()
            )

            def coverage(
                count
            ):

                if total_customers == 0:

                    return 0.0

                return float(
                    count
                    /
                    total_customers
                    *
                    100
                )

            summary = {

                "total_customers":
                    total_customers,

                "churn": {

                    "customers":
                        churn_customers,

                    "coverage_percentage":
                        coverage(
                            churn_customers
                        )
                },

                "clv": {

                    "customers":
                        clv_customers,

                    "coverage_percentage":
                        coverage(
                            clv_customers
                        )
                },

                "recommendation": {

                    "customers":
                        recommendation_customers,

                    "coverage_percentage":
                        coverage(
                            recommendation_customers
                        )
                }
            }

            return summary

        except Exception as e:

            logger.error(
                "Customer intelligence generation "
                "summary failed."
            )

            raise CustomException(
                e,
                sys
            )