import json
import os
import sys

import numpy as np
import pandas as pd

from src.utils.recommendation_persistence import (
    RecommendationPersistence
)

from src.utils.exception import CustomException
from src.utils.logger import logger


class RecommendationPredictor:

    def __init__(
        self,
        model_path: str =
            "models/recommendation/recommender.pkl",

        metadata_path: str =
            "models/recommendation/recommender_metadata.json"
    ):

        try:

            logger.info(
                "Initializing Recommendation Predictor."
            )

            self.model_path = (
                model_path
            )

            self.metadata_path = (
                metadata_path
            )

            # =================================
            # LOAD MODEL ONCE
            # =================================

            persistence = (
                RecommendationPersistence()
            )

            self.model = (
                persistence.load_model(
                    self.model_path
                )
            )

            # =================================
            # LOAD METADATA
            # =================================

            self.metadata = (
                self._load_metadata(
                    self.metadata_path
                )
            )

            self.default_top_k = (
                self.metadata
                .get(
                    "prediction",
                    {}
                )
                .get(
                    "default_top_k",
                    10
                )
            )

            self.selected_model = (
                self.metadata
                .get(
                    "selected_model",
                    self.model.__class__.__name__
                )
            )

            logger.info(
                f"Recommendation Predictor initialized. "
                f"Model: {self.selected_model}"
            )

        except Exception as e:

            logger.error(
                "Recommendation Predictor "
                "initialization failed."
            )

            raise CustomException(
                e,
                sys
            )

    def recommend(
        self,
        customer_id,
        top_k: int = None,
        mode: str = "next_purchase"
    ) -> dict:

        try:

            # =================================
            # VALIDATE INPUT
            # =================================

            if customer_id is None:

                raise ValueError(
                    "customer_id cannot be None."
                )

            if top_k is None:

                top_k = (
                    self.default_top_k
                )

            if not isinstance(
                top_k,
                int
            ):

                raise ValueError(
                    "top_k must be an integer."
                )

            if top_k < 1:

                raise ValueError(
                    "top_k must be at least 1."
                )

            valid_modes = [
                "next_purchase",
                "discovery"
            ]

            if (
                mode
                not in
                valid_modes
            ):

                raise ValueError(
                    f"Unsupported recommendation mode: "
                    f"{mode}. "
                    f"Allowed modes: {valid_modes}"
                )

            # =================================
            # MODE CONFIGURATION
            # =================================

            exclude_already_purchased = (
                mode
                ==
                "discovery"
            )

            # =================================
            # DETECT KNOWN CUSTOMER
            # =================================

            known_customer = (
                self._is_known_customer(
                    customer_id
                )
            )

            # =================================
            # GET RECOMMENDATIONS
            # =================================

            recommendations = (
                self.model.recommend(
                    customer_id=
                        customer_id,

                    top_k=
                        top_k,

                    exclude_already_purchased=
                        exclude_already_purchased
                )
            )

            # =================================
            # FALLBACK IF NEEDED
            # =================================

            recommendation_source = (
                self.selected_model
            )

            if recommendations.empty:

                fallback = (
                    self._popularity_fallback(
                        customer_id=
                            customer_id,

                        top_k=
                            top_k,

                        exclude_already_purchased=
                            exclude_already_purchased
                    )
                )

                if not fallback.empty:

                    recommendations = (
                        fallback
                    )

                    recommendation_source = (
                        "Popularity Fallback"
                    )

            # =================================
            # STANDARDIZE OUTPUT
            # =================================

            standardized = (
                self._standardize_recommendations(
                    recommendations
                )
            )

            result = {

                "customer_id":
                    self._json_safe_value(
                        customer_id
                    ),

                "known_customer":
                    bool(
                        known_customer
                    ),

                "mode":
                    mode,

                "exclude_already_purchased":
                    bool(
                        exclude_already_purchased
                    ),

                "requested_top_k":
                    int(
                        top_k
                    ),

                "recommendation_count":
                    int(
                        len(
                            standardized
                        )
                    ),

                "recommendation_source":
                    recommendation_source,

                "recommendations":
                    standardized
            }

            logger.info(
                f"Recommendation completed for "
                f"customer {customer_id}. "
                f"Count: {len(standardized)}"
            )

            return result

        except Exception as e:

            logger.error(
                "Recommendation prediction failed."
            )

            raise CustomException(
                e,
                sys
            )

    def recommend_batch(
        self,
        customer_ids,
        top_k: int = None,
        mode: str = "next_purchase"
    ) -> pd.DataFrame:

        try:

            # =================================
            # CUSTOMER INPUT
            # =================================

            if isinstance(
                customer_ids,
                pd.Series
            ):

                customers = (
                    customer_ids
                    .tolist()
                )

            elif isinstance(
                customer_ids,
                np.ndarray
            ):

                customers = (
                    customer_ids
                    .tolist()
                )

            elif isinstance(
                customer_ids,
                (
                    list,
                    tuple,
                    set
                )
            ):

                customers = list(
                    customer_ids
                )

            else:

                raise ValueError(
                    "customer_ids must be a list, tuple, "
                    "set, pandas Series, or numpy array."
                )

            if len(customers) == 0:

                raise ValueError(
                    "customer_ids cannot be empty."
                )

            # =================================
            # UNIQUE CUSTOMERS
            # =================================

            customers = list(
                dict.fromkeys(
                    customers
                )
            )

            rows = []

            # =================================
            # PREDICT EACH CUSTOMER
            # =================================

            for customer_id in customers:

                result = (
                    self.recommend(
                        customer_id=
                            customer_id,

                        top_k=
                            top_k,

                        mode=
                            mode
                    )
                )

                for recommendation in (
                    result[
                        "recommendations"
                    ]
                ):

                    rows.append(
                        {
                            "CustomerID":
                                result[
                                    "customer_id"
                                ],

                            "KnownCustomer":
                                result[
                                    "known_customer"
                                ],

                            "Mode":
                                result[
                                    "mode"
                                ],

                            "RecommendationSource":
                                result[
                                    "recommendation_source"
                                ],

                            "StockCode":
                                recommendation[
                                    "StockCode"
                                ],

                            "Description":
                                recommendation[
                                    "Description"
                                ],

                            "Score":
                                recommendation[
                                    "Score"
                                ],

                            "Rank":
                                recommendation[
                                    "Rank"
                                ]
                        }
                    )

            return pd.DataFrame(
                rows,
                columns=[
                    "CustomerID",
                    "KnownCustomer",
                    "Mode",
                    "RecommendationSource",
                    "StockCode",
                    "Description",
                    "Score",
                    "Rank"
                ]
            )

        except Exception as e:

            logger.error(
                "Batch recommendation prediction failed."
            )

            raise CustomException(
                e,
                sys
            )

    def _is_known_customer(
        self,
        customer_id
    ) -> bool:

        # ---------------------------------
        # Most recommendation models have
        # customer_purchases.
        # ---------------------------------

        if hasattr(
            self.model,
            "customer_purchases"
        ):

            return (
                customer_id
                in
                self.model.customer_purchases
            )

        # ---------------------------------
        # Matrix factorization / Item-CF
        # may expose customer_to_index.
        # ---------------------------------

        if hasattr(
            self.model,
            "customer_to_index"
        ):

            return (
                customer_id
                in
                self.model.customer_to_index
            )

        # ---------------------------------
        # Hybrid nested customer history
        # ---------------------------------

        if hasattr(
            self.model,
            "item_cf_model"
        ):

            item_model = (
                self.model.item_cf_model
            )

            if hasattr(
                item_model,
                "customer_to_index"
            ):

                return (
                    customer_id
                    in
                    item_model.customer_to_index
                )

        return False

    def _popularity_fallback(
        self,
        customer_id,
        top_k: int,
        exclude_already_purchased: bool
    ) -> pd.DataFrame:

        try:

            # =================================
            # DIRECT POPULARITY MODEL
            # =================================

            if (
                self.model.__class__.__name__
                ==
                "PopularityRecommender"
            ):

                return (
                    self.model.recommend(
                        customer_id=
                            customer_id,

                        top_k=
                            top_k,

                        exclude_already_purchased=
                            exclude_already_purchased
                    )
                )

            # =================================
            # HYBRID CONTAINS POPULARITY
            # =================================

            if hasattr(
                self.model,
                "popularity_model"
            ):

                return (
                    self.model
                    .popularity_model
                    .recommend(
                        customer_id=
                            customer_id,

                        top_k=
                            top_k,

                        exclude_already_purchased=
                            exclude_already_purchased
                    )
                )

            # =================================
            # NO POPULARITY COMPONENT
            # =================================

            logger.warning(
                "No popularity fallback is available "
                "inside the selected recommender."
            )

            return pd.DataFrame()

        except Exception as e:

            logger.error(
                "Popularity fallback failed."
            )

            raise CustomException(
                e,
                sys
            )

    def _standardize_recommendations(
        self,
        recommendations: pd.DataFrame
    ) -> list:

        if (
            recommendations is None
            or
            recommendations.empty
        ):

            return []

        data = (
            recommendations
            .copy()
        )

        # =================================
        # PRODUCT ID
        # =================================

        if (
            "StockCode"
            not in
            data.columns
        ):

            raise ValueError(
                "Recommendation output does not "
                "contain StockCode."
            )

        # =================================
        # DESCRIPTION
        # =================================

        if (
            "Description"
            not in
            data.columns
        ):

            data[
                "Description"
            ] = ""

        # =================================
        # DETECT SCORE COLUMN
        # =================================

        score_candidates = [
            "HybridScore",
            "RecommendationScore",
            "PopularityScore"
        ]

        score_column = None

        for candidate in score_candidates:

            if candidate in data.columns:

                score_column = (
                    candidate
                )

                break

        if score_column is None:

            data[
                "_Score"
            ] = 0.0

            score_column = (
                "_Score"
            )

        # =================================
        # DETECT RANK COLUMN
        # =================================

        if (
            "RecommendationRank"
            in data.columns
        ):

            rank_values = (
                data[
                    "RecommendationRank"
                ]
            )

        elif (
            "PopularityRank"
            in data.columns
        ):

            rank_values = (
                data[
                    "PopularityRank"
                ]
            )

        else:

            rank_values = pd.Series(
                np.arange(
                    1,
                    len(data) + 1
                ),
                index=data.index
            )

        # =================================
        # JSON-FRIENDLY RECORDS
        # =================================

        results = []

        for position, (
            index,
            row
        ) in enumerate(
            data.iterrows()
        ):

            score = (
                row[
                    score_column
                ]
            )

            if pd.isna(
                score
            ):

                score = 0.0

            rank = (
                rank_values.loc[
                    index
                ]
            )

            results.append(
                {
                    "StockCode":
                        str(
                            row[
                                "StockCode"
                            ]
                        ),

                    "Description":
                        str(
                            row[
                                "Description"
                            ]
                        ),

                    "Score":
                        float(
                            score
                        ),

                    "Rank":
                        int(
                            rank
                        )
                }
            )

        # ---------------------------------
        # Always make API rank sequential
        # ---------------------------------

        results = sorted(
            results,
            key=lambda item:
                item[
                    "Rank"
                ]
        )

        for position, item in enumerate(
            results,
            start=1
        ):

            item[
                "Rank"
            ] = position

        return results

    def _load_metadata(
        self,
        file_path: str
    ) -> dict:

        if not os.path.exists(
            file_path
        ):

            raise FileNotFoundError(
                f"Recommendation metadata "
                f"not found: {file_path}"
            )

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )

    def _json_safe_value(
        self,
        value
    ):

        if hasattr(
            value,
            "item"
        ):

            return value.item()

        return value