import sys

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.utils.exception import CustomException
from src.utils.logger import logger


class MatrixFactorizationRecommender:

    def __init__(
        self,
        n_components: int = 20,
        random_state: int = 42
    ):

        self.n_components = (
            n_components
        )

        self.random_state = (
            random_state
        )

        self.model = None

        self.customer_item_matrix = None

        self.user_factors = None
        self.item_factors = None

        self.customer_ids = []
        self.product_ids = []

        self.customer_to_index = {}
        self.product_to_index = {}
        self.index_to_product = {}

        self.customer_purchases = {}
        self.product_descriptions = {}

        self.catalog_products = set()

        self.actual_n_components = None

    def fit(
        self,
        interactions: pd.DataFrame
    ):

        try:

            logger.info(
                "Training matrix factorization recommender."
            )

            # ---------------------------------
            # Validate required columns
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
                    f"Missing matrix factorization columns: "
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
            # Standardize product IDs
            # ---------------------------------

            data[
                "StockCode"
            ] = (
                data[
                    "StockCode"
                ]
                .astype(str)
                .str.strip()
            )

            # ---------------------------------
            # Numeric interaction validation
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
                    "InteractionStrength contains "
                    "missing or invalid values."
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

            # =================================
            # CUSTOMER / PRODUCT MAPPINGS
            # =================================

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
            # Product descriptions
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
            # SPARSE CUSTOMER-ITEM MATRIX
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
            # COMPONENT VALIDATION
            # =================================

            maximum_components = (
                min(
                    self.customer_item_matrix.shape
                )
                - 1
            )

            if maximum_components < 1:

                raise ValueError(
                    "Matrix factorization requires "
                    "at least two customers and "
                    "two products."
                )

            self.actual_n_components = min(
                self.n_components,
                maximum_components
            )

            logger.info(
                f"Requested latent factors: "
                f"{self.n_components}"
            )

            logger.info(
                f"Actual latent factors used: "
                f"{self.actual_n_components}"
            )

            # =================================
            # TRAIN TRUNCATED SVD
            # =================================

            self.model = TruncatedSVD(
                n_components=
                    self.actual_n_components,

                random_state=
                    self.random_state
            )

            # ---------------------------------
            # Customer latent vectors
            #
            # Shape:
            # customers × latent factors
            # ---------------------------------

            self.user_factors = (
                self.model.fit_transform(
                    self.customer_item_matrix
                )
            )

            # ---------------------------------
            # Product latent vectors
            #
            # Shape:
            # latent factors × products
            # ---------------------------------

            self.item_factors = (
                self.model.components_
            )

            logger.info(
                "Matrix factorization recommender "
                "trained successfully."
            )

            logger.info(
                f"Customers: "
                f"{len(self.customer_ids)}"
            )

            logger.info(
                f"Products: "
                f"{len(self.product_ids)}"
            )

            logger.info(
                f"Latent factors: "
                f"{self.actual_n_components}"
            )

            logger.info(
                f"Explained variance: "
                f"{self.model.explained_variance_ratio_.sum():.4f}"
            )

            return self

        except Exception as e:

            logger.error(
                "Matrix factorization training failed."
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

            if self.model is None:

                raise ValueError(
                    "Matrix factorization recommender "
                    "has not been fitted."
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

            # =================================
            # RECONSTRUCT PREFERENCE SCORES
            # =================================
            #
            # User latent vector:
            #
            # 1 × latent factors
            #
            # Item factor matrix:
            #
            # latent factors × products
            #
            # Result:
            #
            # 1 × products
            # =================================

            customer_latent_vector = (
                self.user_factors[
                    customer_index
                ]
            )

            scores = (
                customer_latent_vector
                @
                self.item_factors
            )

            scores = np.asarray(
                scores,
                dtype=float
            ).ravel()

            # ---------------------------------
            # Discovery mode
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
            # Sort descending
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
                ):

                    continue

                # ---------------------------------
                # Only positive preference scores
                # ---------------------------------

                if score <= 0:

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
                "Matrix factorization recommendation failed."
            )

            raise CustomException(
                e,
                sys
            )