import sys

import numpy as np
import pandas as pd

from src.recommendation.popularity_recommender import (
    PopularityRecommender
)

from src.recommendation.item_based_cf import (
    ItemBasedCollaborativeRecommender
)

from src.recommendation.matrix_factorization import (
    MatrixFactorizationRecommender
)

from src.utils.exception import CustomException
from src.utils.logger import logger


class HybridRecommender:

    def __init__(
        self,
        item_cf_weight: float = 0.40,
        matrix_factorization_weight: float = 0.40,
        popularity_weight: float = 0.20,
        n_components: int = 20,
        random_state: int = 42,
        candidate_multiplier: int = 5
    ):

        try:

            self.item_cf_weight = (
                item_cf_weight
            )

            self.matrix_factorization_weight = (
                matrix_factorization_weight
            )

            self.popularity_weight = (
                popularity_weight
            )

            self.n_components = (
                n_components
            )

            self.random_state = (
                random_state
            )

            self.candidate_multiplier = (
                candidate_multiplier
            )

            # ---------------------------------
            # Validate configuration
            # ---------------------------------

            weights = [
                self.item_cf_weight,
                self.matrix_factorization_weight,
                self.popularity_weight
            ]

            if any(
                weight < 0
                for weight in weights
            ):

                raise ValueError(
                    "Hybrid recommendation weights "
                    "cannot be negative."
                )

            total_weight = sum(
                weights
            )

            if total_weight <= 0:

                raise ValueError(
                    "At least one hybrid recommendation "
                    "weight must be greater than zero."
                )

            if self.candidate_multiplier < 1:

                raise ValueError(
                    "candidate_multiplier must be at least 1."
                )

            # ---------------------------------
            # Normalize weights
            #
            # Example:
            #
            # 4, 4, 2
            #
            # becomes:
            #
            # 0.4, 0.4, 0.2
            # ---------------------------------

            self.item_cf_weight = (
                self.item_cf_weight
                /
                total_weight
            )

            self.matrix_factorization_weight = (
                self.matrix_factorization_weight
                /
                total_weight
            )

            self.popularity_weight = (
                self.popularity_weight
                /
                total_weight
            )

            self.popularity_model = (
                PopularityRecommender()
            )

            self.item_cf_model = (
                ItemBasedCollaborativeRecommender()
            )

            self.matrix_factorization_model = (
                MatrixFactorizationRecommender(
                    n_components=
                        self.n_components,

                    random_state=
                        self.random_state
                )
            )

            self.catalog_products = set()

            self.customer_purchases = {}

            self.is_fitted = False

        except Exception as e:

            raise CustomException(
                e,
                sys
            )

    def fit(
        self,
        interactions: pd.DataFrame
    ):

        try:

            logger.info(
                "Training hybrid recommendation system."
            )

            if interactions.empty:

                raise ValueError(
                    "Hybrid training interactions are empty."
                )

            # =================================
            # TRAIN COMPONENT MODELS
            # =================================

            self.popularity_model.fit(
                interactions
            )

            self.item_cf_model.fit(
                interactions
            )

            self.matrix_factorization_model.fit(
                interactions
            )

            # ---------------------------------
            # Shared product catalog
            # ---------------------------------

            self.catalog_products = set(
                self.popularity_model
                .catalog_products
            )

            # ---------------------------------
            # Shared purchase history
            # ---------------------------------

            self.customer_purchases = (
                self.item_cf_model
                .customer_purchases
            )

            self.is_fitted = True

            logger.info(
                "Hybrid recommendation system "
                "trained successfully."
            )

            logger.info(
                f"Item-CF Weight: "
                f"{self.item_cf_weight:.2f}"
            )

            logger.info(
                f"Matrix Factorization Weight: "
                f"{self.matrix_factorization_weight:.2f}"
            )

            logger.info(
                f"Popularity Weight: "
                f"{self.popularity_weight:.2f}"
            )

            return self

        except Exception as e:

            logger.error(
                "Hybrid recommender training failed."
            )

            raise CustomException(
                e,
                sys
            )

    def recommend(
        self,
        customer_id,
        top_k: int = 10,
        exclude_already_purchased: bool = False
    ) -> pd.DataFrame:

        try:

            if not self.is_fitted:

                raise ValueError(
                    "Hybrid recommender has not been fitted."
                )

            if top_k < 1:

                raise ValueError(
                    "top_k must be at least 1."
                )

            # =================================
            # CANDIDATE POOL SIZE
            # =================================

            candidate_k = max(
                top_k
                *
                self.candidate_multiplier,
                top_k
            )

            # =================================
            # POPULARITY CANDIDATES
            # =================================

            popularity_recommendations = (
                self.popularity_model
                .recommend(
                    customer_id=
                        customer_id,

                    top_k=
                        candidate_k,

                    exclude_already_purchased=
                        exclude_already_purchased
                )
            )

            # =================================
            # ITEM-CF CANDIDATES
            # =================================

            item_cf_recommendations = (
                self.item_cf_model
                .recommend(
                    customer_id=
                        customer_id,

                    top_k=
                        candidate_k,

                    exclude_already_purchased=
                        exclude_already_purchased
                )
            )

            # =================================
            # MATRIX FACTORIZATION CANDIDATES
            # =================================

            matrix_recommendations = (
                self.matrix_factorization_model
                .recommend(
                    customer_id=
                        customer_id,

                    top_k=
                        candidate_k,

                    exclude_already_purchased=
                        exclude_already_purchased
                )
            )

            # =================================
            # COLD START
            # =================================
            #
            # Unknown customers will have:
            #
            # Item-CF = empty
            # Matrix Factorization = empty
            #
            # Popularity remains available.
            # =================================

            if (
                item_cf_recommendations.empty
                and
                matrix_recommendations.empty
            ):

                logger.info(
                    f"Cold-start customer detected: "
                    f"{customer_id}. "
                    f"Using popularity fallback."
                )

                fallback = (
                    popularity_recommendations
                    .head(
                        top_k
                    )
                    .copy()
                )

                if fallback.empty:

                    return pd.DataFrame(
                        columns=[
                            "StockCode",
                            "Description",
                            "HybridScore",
                            "ItemCFScore",
                            "MatrixFactorizationScore",
                            "PopularityScore",
                            "RecommendationRank",
                            "RecommendationSource"
                        ]
                    )

                fallback[
                    "HybridScore"
                ] = (
                    self._normalize_scores(
                        fallback[
                            "PopularityScore"
                        ]
                    )
                )

                fallback[
                    "ItemCFScore"
                ] = 0.0

                fallback[
                    "MatrixFactorizationScore"
                ] = 0.0

                fallback[
                    "RecommendationSource"
                ] = "Popularity Fallback"

                fallback[
                    "RecommendationRank"
                ] = np.arange(
                    1,
                    len(fallback) + 1
                )

                return fallback[
                    [
                        "StockCode",
                        "Description",
                        "HybridScore",
                        "ItemCFScore",
                        "MatrixFactorizationScore",
                        "PopularityScore",
                        "RecommendationRank",
                        "RecommendationSource"
                    ]
                ]

            # =================================
            # PREPARE COMPONENT SCORES
            # =================================

            popularity_scores = (
                self._prepare_component_scores(
                    recommendations=
                        popularity_recommendations,

                    score_column=
                        "PopularityScore",

                    output_column=
                        "PopularityScore"
                )
            )

            item_cf_scores = (
                self._prepare_component_scores(
                    recommendations=
                        item_cf_recommendations,

                    score_column=
                        "RecommendationScore",

                    output_column=
                        "ItemCFScore"
                )
            )

            matrix_scores = (
                self._prepare_component_scores(
                    recommendations=
                        matrix_recommendations,

                    score_column=
                        "RecommendationScore",

                    output_column=
                        "MatrixFactorizationScore"
                )
            )

            # =================================
            # BUILD PRODUCT CANDIDATE POOL
            # =================================

            candidate_products = set()

            candidate_products.update(
                popularity_scores.keys()
            )

            candidate_products.update(
                item_cf_scores.keys()
            )

            candidate_products.update(
                matrix_scores.keys()
            )

            # ---------------------------------
            # Description lookup
            # ---------------------------------

            description_lookup = dict(
                self.popularity_model
                .product_ranking[
                    [
                        "StockCode",
                        "Description"
                    ]
                ]
                .astype(
                    {
                        "StockCode":
                            str
                    }
                )
                .values
            )

            # =================================
            # HYBRID SCORING
            # =================================

            results = []

            for product_id in candidate_products:

                popularity_score = (
                    popularity_scores.get(
                        product_id,
                        0.0
                    )
                )

                item_cf_score = (
                    item_cf_scores.get(
                        product_id,
                        0.0
                    )
                )

                matrix_score = (
                    matrix_scores.get(
                        product_id,
                        0.0
                    )
                )

                hybrid_score = (

                    self.item_cf_weight
                    *
                    item_cf_score

                    +

                    self.matrix_factorization_weight
                    *
                    matrix_score

                    +

                    self.popularity_weight
                    *
                    popularity_score
                )

                if hybrid_score <= 0:

                    continue

                results.append(
                    {
                        "StockCode":
                            product_id,

                        "Description":
                            description_lookup.get(
                                product_id,
                                ""
                            ),

                        "HybridScore":
                            float(
                                hybrid_score
                            ),

                        "ItemCFScore":
                            float(
                                item_cf_score
                            ),

                        "MatrixFactorizationScore":
                            float(
                                matrix_score
                            ),

                        "PopularityScore":
                            float(
                                popularity_score
                            ),

                        "RecommendationSource":
                            "Hybrid"
                    }
                )

            # =================================
            # FINAL RANKING
            # =================================

            result = pd.DataFrame(
                results
            )

            if result.empty:

                return pd.DataFrame(
                    columns=[
                        "StockCode",
                        "Description",
                        "HybridScore",
                        "ItemCFScore",
                        "MatrixFactorizationScore",
                        "PopularityScore",
                        "RecommendationRank",
                        "RecommendationSource"
                    ]
                )

            result = (
                result
                .sort_values(
                    by="HybridScore",
                    ascending=False
                )
                .head(
                    top_k
                )
                .reset_index(
                    drop=True
                )
            )

            result[
                "RecommendationRank"
            ] = np.arange(
                1,
                len(result) + 1
            )

            return result[
                [
                    "StockCode",
                    "Description",
                    "HybridScore",
                    "ItemCFScore",
                    "MatrixFactorizationScore",
                    "PopularityScore",
                    "RecommendationRank",
                    "RecommendationSource"
                ]
            ]

        except Exception as e:

            logger.error(
                "Hybrid recommendation failed."
            )

            raise CustomException(
                e,
                sys
            )

    def _prepare_component_scores(
        self,
        recommendations: pd.DataFrame,
        score_column: str,
        output_column: str
    ) -> dict:

        if recommendations.empty:

            return {}

        data = (
            recommendations[
                [
                    "StockCode",
                    score_column
                ]
            ]
            .copy()
        )

        data[
            "StockCode"
        ] = (
            data[
                "StockCode"
            ]
            .astype(str)
        )

        data[
            output_column
        ] = (
            self._normalize_scores(
                data[
                    score_column
                ]
            )
        )

        return dict(
            zip(
                data[
                    "StockCode"
                ],
                data[
                    output_column
                ]
            )
        )

    def _normalize_scores(
        self,
        scores
    ):

        values = np.asarray(
            scores,
            dtype=float
        )

        if len(values) == 0:

            return values

        values[
            ~np.isfinite(
                values
            )
        ] = 0.0

        maximum = float(
            np.max(
                values
            )
        )

        minimum = float(
            np.min(
                values
            )
        )

        # ---------------------------------
        # Normal case: Min-Max scaling
        # ---------------------------------

        if maximum > minimum:

            return (
                (
                    values
                    -
                    minimum
                )
                /
                (
                    maximum
                    -
                    minimum
                )
            )

        # ---------------------------------
        # All positive values equal
        # ---------------------------------

        if maximum > 0:

            return np.ones(
                len(values),
                dtype=float
            )

        return np.zeros(
            len(values),
            dtype=float
        )