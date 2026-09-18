import sys

import pandas as pd

from sklearn.cluster import KMeans

from src.utils.exception import CustomException
from src.utils.logger import logger


class CustomerSegmentationModel:

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: int = 42
    ):

        self.n_clusters = n_clusters
        self.random_state = random_state

    def build_model(
        self
    ) -> KMeans:

        try:

            logger.info(
                "Building K-Means clustering model."
            )

            model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )

            return model

        except Exception as e:

            logger.error(
                "K-Means model creation failed."
            )

            raise CustomException(
                e,
                sys
            )

    def train(
        self,
        X: pd.DataFrame
    ) -> KMeans:

        try:

            logger.info(
                "Starting K-Means model training."
            )

            if X.empty:

                raise ValueError(
                    "Segmentation feature dataset is empty."
                )

            model = self.build_model()

            model.fit(
                X
            )

            logger.info(
                "K-Means model training completed."
            )

            logger.info(
                f"Final inertia: "
                f"{model.inertia_}"
            )

            return model

        except Exception as e:

            logger.error(
                "K-Means model training failed."
            )

            raise CustomException(
                e,
                sys
            )

    def assign_clusters(
        self,
        model: KMeans,
        X: pd.DataFrame
    ) -> pd.Series:

        try:

            logger.info(
                "Assigning customers to clusters."
            )

            labels = model.predict(
                X
            )

            cluster_series = pd.Series(
                labels,
                index=X.index,
                name="Cluster"
            )

            logger.info(
                "Customer cluster assignment completed."
            )

            return cluster_series

        except Exception as e:

            logger.error(
                "Customer cluster assignment failed."
            )

            raise CustomException(
                e,
                sys
            )