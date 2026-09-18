import sys

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.utils.exception import CustomException
from src.utils.logger import logger


class SegmentationClusterAnalysis:

    def evaluate_clusters(
        self,
        X: pd.DataFrame,
        min_clusters: int = 2,
        max_clusters: int = 8
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Starting segmentation cluster analysis."
            )

            if X.empty:

                raise ValueError(
                    "Segmentation feature dataset is empty."
                )

            if min_clusters < 2:

                raise ValueError(
                    "Minimum number of clusters must be at least 2."
                )

            if max_clusters <= min_clusters:

                raise ValueError(
                    "Maximum clusters must be greater than minimum clusters."
                )

            if max_clusters >= len(X):

                raise ValueError(
                    "Maximum clusters must be smaller than the number of customers."
                )

            results = []

            for k in range(
                min_clusters,
                max_clusters + 1
            ):

                logger.info(
                    f"Evaluating KMeans with k={k}."
                )

                model = KMeans(
                    n_clusters=k,
                    random_state=42,
                    n_init=10
                )

                labels = model.fit_predict(
                    X
                )

                inertia = float(
                    model.inertia_
                )

                silhouette = float(
                    silhouette_score(
                        X,
                        labels
                    )
                )

                cluster_counts = (
                    pd.Series(labels)
                    .value_counts()
                    .sort_index()
                )

                smallest_cluster = int(
                    cluster_counts.min()
                )

                largest_cluster = int(
                    cluster_counts.max()
                )

                results.append(
                    {
                        "K":
                            k,

                        "Inertia":
                            inertia,

                        "SilhouetteScore":
                            silhouette,

                        "SmallestCluster":
                            smallest_cluster,

                        "LargestCluster":
                            largest_cluster
                    }
                )

            report = pd.DataFrame(
                results
            )

            logger.info(
                f"Segmentation cluster analysis completed:\n"
                f"{report}"
            )

            return report

        except Exception as e:

            logger.error(
                "Segmentation cluster analysis failed."
            )

            raise CustomException(
                e,
                sys
            )