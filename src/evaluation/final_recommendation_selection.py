import sys

import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class FinalRecommendationSelector:

    def __init__(
        self,
        selection_k: int = 10
    ):

        self.selection_k = (
            selection_k
        )

    def select_model(
        self,
        comparison: pd.DataFrame
    ) -> dict:

        try:

            logger.info(
                "Starting final recommendation "
                "model selection."
            )

            # ---------------------------------
            # Validate input
            # ---------------------------------

            required_columns = [
                "Model",
                "K",
                "PrecisionAtK",
                "RecallAtK",
                "HitRateAtK",
                "CatalogCoverage",
                "UniqueRecommendedProducts"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in comparison.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing recommendation comparison "
                    f"columns: {missing_columns}"
                )

            if comparison.empty:

                raise ValueError(
                    "Recommendation comparison "
                    "dataset is empty."
                )

            # ---------------------------------
            # Select requested K
            # ---------------------------------

            selection_data = (
                comparison[
                    comparison[
                        "K"
                    ]
                    ==
                    self.selection_k
                ]
                .copy()
            )

            if selection_data.empty:

                raise ValueError(
                    f"No recommendation results "
                    f"found for K={self.selection_k}."
                )

            # ---------------------------------
            # Validate metrics
            # ---------------------------------

            metric_columns = [
                "PrecisionAtK",
                "RecallAtK",
                "HitRateAtK",
                "CatalogCoverage"
            ]

            if (
                selection_data[
                    metric_columns
                ]
                .isnull()
                .any()
                .any()
            ):

                raise ValueError(
                    "Recommendation comparison "
                    "contains missing metrics."
                )

            # =================================
            # MODEL SELECTION
            # =================================
            #
            # Primary:
            # Recall@K descending
            #
            # Tie breaker 1:
            # HitRate@K descending
            #
            # Tie breaker 2:
            # Precision@K descending
            #
            # Tie breaker 3:
            # Coverage descending
            # =================================

            ranked = (
                selection_data
                .sort_values(
                    by=[
                        "RecallAtK",
                        "HitRateAtK",
                        "PrecisionAtK",
                        "CatalogCoverage",
                        "Model"
                    ],
                    ascending=[
                        False,
                        False,
                        False,
                        False,
                        True
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            # ---------------------------------
            # Add selection rank
            # ---------------------------------

            ranked[
                "SelectionRank"
            ] = (
                ranked.index
                +
                1
            )

            winner = (
                ranked.iloc[0]
            )

            result = {

                "selected_model":
                    str(
                        winner[
                            "Model"
                        ]
                    ),

                "selection_k":
                    int(
                        self.selection_k
                    ),

                "selection_rule": {
                    "primary_metric":
                        f"Recall@{self.selection_k}",

                    "tie_breaker_1":
                        f"HitRate@{self.selection_k}",

                    "tie_breaker_2":
                        f"Precision@{self.selection_k}",

                    "tie_breaker_3":
                        "CatalogCoverage"
                },

                "selected_metrics": {

                    "precision_at_k":
                        float(
                            winner[
                                "PrecisionAtK"
                            ]
                        ),

                    "recall_at_k":
                        float(
                            winner[
                                "RecallAtK"
                            ]
                        ),

                    "hit_rate_at_k":
                        float(
                            winner[
                                "HitRateAtK"
                            ]
                        ),

                    "catalog_coverage":
                        float(
                            winner[
                                "CatalogCoverage"
                            ]
                        ),

                    "unique_recommended_products":
                        int(
                            winner[
                                "UniqueRecommendedProducts"
                            ]
                        )
                },

                "ranked_models":
                    ranked.to_dict(
                        orient="records"
                    )
            }

            logger.info(
                f"Selected recommendation model: "
                f"{result['selected_model']}"
            )

            logger.info(
                f"Recall@{self.selection_k}: "
                f"{result['selected_metrics']['recall_at_k']:.4f}"
            )

            return result

        except Exception as e:

            logger.error(
                "Final recommendation model "
                "selection failed."
            )

            raise CustomException(
                e,
                sys
            )