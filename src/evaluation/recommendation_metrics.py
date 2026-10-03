import sys

import numpy as np
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class RecommendationMetrics:

    def evaluate(
        self,
        recommender,
        ground_truth: pd.DataFrame,
        catalog_products,
        top_k: int = 10,
        exclude_already_purchased: bool = False
    ) -> dict:

        try:

            logger.info(
                f"Starting recommendation evaluation at K={top_k}."
            )

            if top_k < 1:

                raise ValueError(
                    "top_k must be at least 1."
                )

            required_columns = [
                "CustomerID",
                "StockCode"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in ground_truth.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing recommendation ground truth columns: "
                    f"{missing_columns}"
                )

            if ground_truth.empty:

                raise ValueError(
                    "Recommendation ground truth is empty."
                )

            catalog_products = set(
                str(product)
                for product
                in catalog_products
            )

            if not catalog_products:

                raise ValueError(
                    "Recommendation product catalog is empty."
                )

            # ---------------------------------
            # Ground truth by customer
            # ---------------------------------

            customer_ground_truth = (
                ground_truth
                .assign(
                    StockCode=
                        ground_truth[
                            "StockCode"
                        ]
                        .astype(str)
                )
                .groupby(
                    "CustomerID"
                )[
                    "StockCode"
                ]
                .apply(set)
                .to_dict()
            )

            precision_scores = []
            recall_scores = []
            hit_scores = []

            all_recommended_products = set()

            user_results = []

            # =================================
            # CUSTOMER LOOP
            # =================================

            for (
                customer_id,
                actual_products
            ) in customer_ground_truth.items():

                recommendations = (
                    recommender.recommend(
                        customer_id=
                            customer_id,

                        top_k=
                            top_k,

                        exclude_already_purchased=
                            exclude_already_purchased
                    )
                )

                recommended_products = (
                    recommendations[
                        "StockCode"
                    ]
                    .astype(str)
                    .tolist()
                )

                recommended_set = set(
                    recommended_products
                )

                all_recommended_products.update(
                    recommended_set
                )

                hits = (
                    actual_products
                    .intersection(
                        recommended_set
                    )
                )

                hit_count = len(
                    hits
                )

                # ---------------------------------
                # Precision@K
                # ---------------------------------

                precision = (
                    hit_count
                    /
                    top_k
                )

                # ---------------------------------
                # Recall@K
                # ---------------------------------

                recall = (
                    hit_count
                    /
                    len(
                        actual_products
                    )
                )

                # ---------------------------------
                # HitRate@K
                # ---------------------------------

                hit_rate = (
                    1.0
                    if hit_count > 0
                    else 0.0
                )

                precision_scores.append(
                    precision
                )

                recall_scores.append(
                    recall
                )

                hit_scores.append(
                    hit_rate
                )

                user_results.append(
                    {
                        "CustomerID":
                            customer_id,

                        "GroundTruthCount":
                            len(
                                actual_products
                            ),

                        "RecommendationCount":
                            len(
                                recommended_products
                            ),

                        "HitCount":
                            hit_count,

                        "PrecisionAtK":
                            precision,

                        "RecallAtK":
                            recall,

                        "HitRateAtK":
                            hit_rate
                    }
                )

            # ---------------------------------
            # Catalog Coverage
            # ---------------------------------

            recommended_catalog_products = (
                all_recommended_products
                .intersection(
                    catalog_products
                )
            )

            catalog_coverage = (
                len(
                    recommended_catalog_products
                )
                /
                len(
                    catalog_products
                )
            )

            # ---------------------------------
            # Aggregate metrics
            # ---------------------------------

            metrics = {

                "k":
                    int(
                        top_k
                    ),

                "customer_count":
                    int(
                        len(
                            customer_ground_truth
                        )
                    ),

                "precision_at_k":
                    float(
                        np.mean(
                            precision_scores
                        )
                    ),

                "recall_at_k":
                    float(
                        np.mean(
                            recall_scores
                        )
                    ),

                "hit_rate_at_k":
                    float(
                        np.mean(
                            hit_scores
                        )
                    ),

                "catalog_coverage":
                    float(
                        catalog_coverage
                    ),

                "unique_recommended_products":
                    int(
                        len(
                            recommended_catalog_products
                        )
                    ),

                "catalog_product_count":
                    int(
                        len(
                            catalog_products
                        )
                    )
            }

            user_results_df = pd.DataFrame(
                user_results
            )

            logger.info(
                f"Recommendation evaluation completed. "
                f"Precision@{top_k}: "
                f"{metrics['precision_at_k']:.4f}, "
                f"Recall@{top_k}: "
                f"{metrics['recall_at_k']:.4f}, "
                f"HitRate@{top_k}: "
                f"{metrics['hit_rate_at_k']:.4f}"
            )

            return {
                "metrics":
                    metrics,

                "user_results":
                    user_results_df
            }

        except Exception as e:

            logger.error(
                "Recommendation evaluation failed."
            )

            raise CustomException(
                e,
                sys
            )