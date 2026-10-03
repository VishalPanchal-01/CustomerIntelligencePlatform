import sys

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.exception import CustomException
from src.utils.logger import logger


class ItemBasedCollaborativeRecommender:

    def __init__(
        self
    ):

        self.customer_item_matrix = None

        self.item_similarity_matrix = None

        self.customer_ids = None
        self.product_ids = None

        self.customer_to_index = {}
        self.product_to_index = {}

        self.index_to_product = {}

        self.customer_purchases = {}

        self.product_descriptions = {}

        self.catalog_products = set()

    def fit(
        self,
        interactions: pd.DataFrame
    ):

        try:

            logger.info(
                "Training item-based collaborative recommender."
            )

            # ---------------------------------
            # Validate columns
            # ---------------------------------

            required_columns = [
                "CustomerID",
                "StockCode",
                "Description",
                "InteractionStrength"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in interactions.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing item-CF columns: "
                    f"{missing_columns}"
                )

            if interactions.empty:

                raise ValueError(
                    "Training interaction dataset is empty."
                )

            data = (
                interactions
                .copy()
            )

            # ---------------------------------
            # Standardize identifiers
            # ---------------------------------

            data[
                "StockCode"
            ] = (
                data[
                    "StockCode"
                ]
                .astype(str)
            )

            # ---------------------------------
            # Validate interaction values
            # ---------------------------------

            data[
                "InteractionStrength"
            ] = pd.to_numeric(
                data[
                    "InteractionStrength"
                ],
                errors="coerce"
            )

            if (
                data[
                    "InteractionStrength"
                ]
                .isnull()
                .any()
            ):

                raise ValueError(
                    "InteractionStrength contains invalid values."
                )

            if (
                data[
                    "InteractionStrength"
                ]
                <= 0
            ).any():

                raise ValueError(
                    "InteractionStrength must be positive."
                )

            # ---------------------------------
            # Unique customers/products
            # ---------------------------------

            self.customer_ids = (
                data[
                    "CustomerID"
                ]
                .drop_duplicates()
                .tolist()
            )

            self.product_ids = (
                data[
                    "StockCode"
                ]
                .drop_duplicates()
                .tolist()
            )

            # ---------------------------------
            # Index mappings
            # ---------------------------------

            self.customer_to_index = {
                customer_id:
                    index
                for index, customer_id
                in enumerate(
                    self.customer_ids
                )
            }

            self.product_to_index = {
                product_id:
                    index
                for index, product_id
                in enumerate(
                    self.product_ids
                )
            }

            self.index_to_product = {
                index:
                    product_id
                for product_id, index
                in self.product_to_index.items()
            }

            # ---------------------------------
            # Customer purchase history
            # ---------------------------------

            self.customer_purchases = (
                data
                .groupby(
                    "CustomerID"
                )[
                    "StockCode"
                ]
                .apply(
                    lambda values:
                        set(
                            values.astype(str)
                        )
                )
                .to_dict()
            )

            # ---------------------------------
            # Product description lookup
            # ---------------------------------

            self.product_descriptions = (
                data
                .drop_duplicates(
                    subset=[
                        "StockCode"
                    ]
                )
                .set_index(
                    "StockCode"
                )[
                    "Description"
                ]
                .to_dict()
            )

            self.catalog_products = set(
                self.product_ids
            )

            # =================================
            # BUILD SPARSE MATRIX
            # =================================

            row_indices = (
                data[
                    "CustomerID"
                ]
                .map(
                    self.customer_to_index
                )
                .to_numpy()
            )

            column_indices = (
                data[
                    "StockCode"
                ]
                .map(
                    self.product_to_index
                )
                .to_numpy()
            )

            values = (
                data[
                    "InteractionStrength"
                ]
                .astype(float)
                .to_numpy()
            )

            self.customer_item_matrix = csr_matrix(
                (
                    values,
                    (
                        row_indices,
                        column_indices
                    )
                ),
                shape=(
                    len(
                        self.customer_ids
                    ),
                    len(
                        self.product_ids
                    )
                )
            )

            # =================================
            # ITEM-ITEM SIMILARITY
            # =================================
            #
            # Customer-item matrix:
            #
            # rows    = customers
            # columns = products
            #
            # For item similarity we transpose:
            #
            # rows    = products
            # columns = customers
            # =================================

            item_customer_matrix = (
                self.customer_item_matrix
                .T
            )

            self.item_similarity_matrix = (
                cosine_similarity(
                    item_customer_matrix,
                    dense_output=False
                )
            )

            logger.info(
                f"Item-based recommender trained. "
                f"Customers: {len(self.customer_ids)}, "
                f"Products: {len(self.product_ids)}, "
                f"Interactions: "
                f"{self.customer_item_matrix.nnz}"
            )

            return self

        except Exception as e:

            logger.error(
                "Item-based collaborative recommender "
                "training failed."
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

            if self.customer_item_matrix is None:

                raise ValueError(
                    "Item-based recommender has not been fitted."
                )

            if top_k < 1:

                raise ValueError(
                    "top_k must be at least 1."
                )

            # ---------------------------------
            # Unknown customer
            # ---------------------------------

            if (
                customer_id
                not in
                self.customer_to_index
            ):

                return pd.DataFrame(
                    columns=[
                        "StockCode",
                        "Description",
                        "RecommendationScore",
                        "RecommendationRank"
                    ]
                )

            customer_index = (
                self.customer_to_index[
                    customer_id
                ]
            )

            # ---------------------------------
            # Customer interaction vector
            # ---------------------------------

            customer_vector = (
                self.customer_item_matrix[
                    customer_index
                ]
            )

            # ---------------------------------
            # Score products
            #
            # user_vector × item similarity
            #
            # Products similar to customer's
            # historical products receive
            # higher scores.
            # ---------------------------------

            scores = (
                customer_vector
                @
                self.item_similarity_matrix
            )

            scores = np.asarray(
                scores.toarray()
            ).ravel()

            # ---------------------------------
            # Optional purchased-item removal
            # ---------------------------------

            if exclude_already_purchased:

                purchased_products = (
                    self.customer_purchases.get(
                        customer_id,
                        set()
                    )
                )

                for product_id in purchased_products:

                    product_index = (
                        self.product_to_index.get(
                            product_id
                        )
                    )

                    if product_index is not None:

                        scores[
                            product_index
                        ] = -np.inf

            # ---------------------------------
            # Remove invalid values
            # ---------------------------------

            scores[
                ~np.isfinite(
                    scores
                )
            ] = -np.inf

            # ---------------------------------
            # Highest scores first
            # ---------------------------------

            ranked_indices = (
                np.argsort(
                    scores
                )[
                    ::-1
                ]
            )

            recommendations = []

            for product_index in ranked_indices:

                score = (
                    scores[
                        product_index
                    ]
                )

                if (
                    not np.isfinite(
                        score
                    )
                    or
                    score <= 0
                ):

                    continue

                product_id = (
                    self.index_to_product[
                        product_index
                    ]
                )

                recommendations.append(
                    {
                        "StockCode":
                            product_id,

                        "Description":
                            self.product_descriptions.get(
                                product_id,
                                ""
                            ),

                        "RecommendationScore":
                            float(
                                score
                            )
                    }
                )

                if (
                    len(
                        recommendations
                    )
                    >=
                    top_k
                ):

                    break

            result = pd.DataFrame(
                recommendations
            )

            if result.empty:

                return pd.DataFrame(
                    columns=[
                        "StockCode",
                        "Description",
                        "RecommendationScore",
                        "RecommendationRank"
                    ]
                )

            result[
                "RecommendationRank"
            ] = np.arange(
                1,
                len(result) + 1
            )

            return result

        except Exception as e:

            logger.error(
                "Item-based recommendation failed."
            )

            raise CustomException(
                e,
                sys
            )