import sys

import numpy as np
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class PopularityRecommender:

    def __init__(self):

        self.product_ranking = None
        self.customer_purchases = {}
        self.catalog_products = set()

    def fit(
        self,
        interactions: pd.DataFrame
    ):

        try:

            logger.info(
                "Training popularity recommendation baseline."
            )

            required_columns = [
                "CustomerID",
                "StockCode",
                "Description",
                "PurchaseCount",
                "TotalQuantity",
                "TotalSpend",
                "InteractionStrength"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in interactions.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing popularity recommender columns: "
                    f"{missing_columns}"
                )

            if interactions.empty:

                raise ValueError(
                    "Training interaction dataset is empty."
                )

            data = interactions.copy()

            data[
                "StockCode"
            ] = (
                data[
                    "StockCode"
                ]
                .astype(str)
            )

            # ---------------------------------
            # Save product catalog
            # ---------------------------------

            self.catalog_products = set(
                data[
                    "StockCode"
                ]
                .unique()
            )

            # ---------------------------------
            # Save purchase history
            # for optional filtering later
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
            # Global popularity aggregation
            # ---------------------------------

            product_stats = (
                data
                .groupby(
                    "StockCode",
                    as_index=False
                )
                .agg(
                    Description=(
                        "Description",
                        "first"
                    ),

                    CustomerCount=(
                        "CustomerID",
                        "nunique"
                    ),

                    PurchaseCount=(
                        "PurchaseCount",
                        "sum"
                    ),

                    TotalQuantity=(
                        "TotalQuantity",
                        "sum"
                    ),

                    TotalSpend=(
                        "TotalSpend",
                        "sum"
                    ),

                    InteractionStrength=(
                        "InteractionStrength",
                        "sum"
                    )
                )
            )

            # ---------------------------------
            # Popularity score
            #
            # CustomerCount:
            # broad product adoption
            #
            # PurchaseCount:
            # repeat purchase behavior
            #
            # TotalQuantity:
            # volume
            #
            # log1p prevents extreme products
            # from dominating completely.
            # ---------------------------------

            product_stats[
                "PopularityScore"
            ] = (
                np.log1p(
                    product_stats[
                        "CustomerCount"
                    ]
                )
                +
                np.log1p(
                    product_stats[
                        "PurchaseCount"
                    ]
                )
                +
                np.log1p(
                    product_stats[
                        "TotalQuantity"
                    ]
                )
            )

            # ---------------------------------
            # Rank products
            # ---------------------------------

            product_stats = (
                product_stats
                .sort_values(
                    by=[
                        "PopularityScore",
                        "CustomerCount",
                        "PurchaseCount"
                    ],
                    ascending=[
                        False,
                        False,
                        False
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            product_stats[
                "PopularityRank"
            ] = (
                np.arange(
                    1,
                    len(product_stats) + 1
                )
            )

            self.product_ranking = (
                product_stats
            )

            logger.info(
                f"Popularity recommender trained "
                f"with {len(product_stats)} products."
            )

            return self

        except Exception as e:

            logger.error(
                "Popularity recommender training failed."
            )

            raise CustomException(
                e,
                sys
            )

    def recommend(
        self,
        customer_id=None,
        top_k: int = 10,
        exclude_already_purchased: bool = False
    ) -> pd.DataFrame:

        try:

            if self.product_ranking is None:

                raise ValueError(
                    "Popularity recommender has not been fitted."
                )

            if top_k < 1:

                raise ValueError(
                    "top_k must be at least 1."
                )

            recommendations = (
                self.product_ranking
                .copy()
            )

            # ---------------------------------
            # Optional discovery-mode filter
            # ---------------------------------

            if (
                exclude_already_purchased
                and
                customer_id is not None
            ):

                purchased_products = (
                    self.customer_purchases.get(
                        customer_id,
                        set()
                    )
                )

                recommendations = (
                    recommendations[
                        ~recommendations[
                            "StockCode"
                        ]
                        .isin(
                            purchased_products
                        )
                    ]
                    .copy()
                )

            recommendations = (
                recommendations
                .head(
                    top_k
                )
                .reset_index(
                    drop=True
                )
            )

            return recommendations

        except Exception as e:

            logger.error(
                "Popularity recommendation failed."
            )

            raise CustomException(
                e,
                sys
            )