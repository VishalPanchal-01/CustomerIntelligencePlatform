import sys

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

from src.recommendation.hybrid_recommender import (
    HybridRecommender
)

from src.utils.exception import CustomException
from src.utils.logger import logger


class FinalRecommendationTrainer:

    def build_model(
        self,
        selected_model: str,
        configuration: dict
    ):

        try:

            logger.info(
                f"Building final recommendation model: "
                f"{selected_model}"
            )

            if (
                selected_model
                ==
                "Popularity Baseline"
            ):

                return (
                    PopularityRecommender()
                )

            if (
                selected_model
                ==
                "Item-Based Collaborative Filtering"
            ):

                return (
                    ItemBasedCollaborativeRecommender()
                )

            if (
                selected_model
                ==
                "Matrix Factorization"
            ):

                return (
                    MatrixFactorizationRecommender(
                        n_components=
                            configuration.get(
                                "n_components",
                                20
                            ),

                        random_state=
                            configuration.get(
                                "random_state",
                                42
                            )
                    )
                )

            if (
                selected_model
                ==
                "Hybrid Recommender"
            ):

                return (
                    HybridRecommender(
                        item_cf_weight=
                            configuration.get(
                                "item_cf_weight",
                                0.40
                            ),

                        matrix_factorization_weight=
                            configuration.get(
                                "matrix_factorization_weight",
                                0.40
                            ),

                        popularity_weight=
                            configuration.get(
                                "popularity_weight",
                                0.20
                            ),

                        n_components=
                            configuration.get(
                                "n_components",
                                20
                            ),

                        random_state=
                            configuration.get(
                                "random_state",
                                42
                            ),

                        candidate_multiplier=
                            configuration.get(
                                "candidate_multiplier",
                                5
                            )
                    )
                )

            raise ValueError(
                f"Unsupported recommendation model: "
                f"{selected_model}"
            )

        except Exception as e:

            logger.error(
                "Final recommendation model "
                "creation failed."
            )

            raise CustomException(
                e,
                sys
            )

    def train(
        self,
        interactions: pd.DataFrame,
        selected_model: str,
        configuration: dict
    ):

        try:

            if interactions.empty:

                raise ValueError(
                    "Production recommendation "
                    "interactions are empty."
                )

            logger.info(
                "Starting final recommendation training."
            )

            model = (
                self.build_model(
                    selected_model=
                        selected_model,

                    configuration=
                        configuration
                )
            )

            model.fit(
                interactions
            )

            logger.info(
                "Final recommendation model "
                "training completed."
            )

            return model

        except Exception as e:

            logger.error(
                "Final recommendation training failed."
            )

            raise CustomException(
                e,
                sys
            )