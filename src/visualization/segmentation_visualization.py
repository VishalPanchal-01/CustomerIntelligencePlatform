import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA

from src.utils.exception import CustomException
from src.utils.logger import logger


class SegmentationVisualization:

    def __init__(
        self,
        output_directory: str = (
            "artifacts/segmentation/plots"
        )
    ):

        self.output_directory = (
            output_directory
        )

        os.makedirs(
            self.output_directory,
            exist_ok=True
        )

        self.features = [
            "Recency",
            "Frequency",
            "Monetary",
            "TotalItems",
            "AverageOrderValue",
            "Tenure"
        ]

    def plot_cluster_distribution(
        self,
        df: pd.DataFrame
    ) -> str:

        try:

            logger.info(
                "Creating cluster distribution plot."
            )

            if "Cluster" not in df.columns:

                raise ValueError(
                    "Cluster column not found."
                )

            cluster_counts = (
                df["Cluster"]
                .value_counts()
                .sort_index()
            )

            plt.figure(
                figsize=(8, 5)
            )

            cluster_counts.plot(
                kind="bar"
            )

            plt.title(
                "Customer Distribution by Cluster"
            )

            plt.xlabel(
                "Cluster"
            )

            plt.ylabel(
                "Number of Customers"
            )

            plt.xticks(
                rotation=0
            )

            plt.tight_layout()

            output_path = os.path.join(
                self.output_directory,
                "cluster_distribution.png"
            )

            plt.savefig(
                output_path,
                bbox_inches="tight"
            )

            plt.close()

            logger.info(
                f"Cluster distribution plot saved to: "
                f"{output_path}"
            )

            return output_path

        except Exception as e:

            logger.error(
                "Cluster distribution plot failed."
            )

            raise CustomException(
                e,
                sys
            )

    def plot_cluster_feature_profile(
        self,
        df: pd.DataFrame
    ) -> str:

        try:

            logger.info(
                "Creating cluster feature profile plot."
            )

            required_columns = (
                self.features
                +
                ["Cluster"]
            )

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing required columns: "
                    f"{missing_columns}"
                )

            cluster_profile = (
                df
                .groupby(
                    "Cluster"
                )[self.features]
                .mean()
            )

            plt.figure(
                figsize=(12, 6)
            )

            cluster_profile.T.plot(
                kind="bar",
                figsize=(12, 6)
            )

            plt.title(
                "Average Feature Profile by Cluster"
            )

            plt.xlabel(
                "Feature"
            )

            plt.ylabel(
                "Average Value"
            )

            plt.xticks(
                rotation=45
            )

            plt.tight_layout()

            output_path = os.path.join(
                self.output_directory,
                "cluster_feature_profile.png"
            )

            plt.savefig(
                output_path,
                bbox_inches="tight"
            )

            plt.close()

            logger.info(
                f"Cluster feature profile saved to: "
                f"{output_path}"
            )

            return output_path

        except Exception as e:

            logger.error(
                "Cluster feature profile plot failed."
            )

            raise CustomException(
                e,
                sys
            )

    def plot_pca_clusters(
        self,
        scaled_features: pd.DataFrame,
        clusters: pd.Series
    ) -> str:

        try:

            logger.info(
                "Creating PCA customer segmentation plot."
            )

            if scaled_features.empty:

                raise ValueError(
                    "Scaled feature dataset is empty."
                )

            if (
                len(scaled_features)
                !=
                len(clusters)
            ):

                raise ValueError(
                    "Feature rows and cluster labels do not match."
                )

            pca = PCA(
                n_components=2
            )

            principal_components = (
                pca.fit_transform(
                    scaled_features
                )
            )

            pca_df = pd.DataFrame(
                principal_components,
                columns=[
                    "PC1",
                    "PC2"
                ],
                index=
                    scaled_features.index
            )

            pca_df[
                "Cluster"
            ] = clusters.to_numpy()

            plt.figure(
                figsize=(10, 7)
            )

            for cluster in sorted(
                pca_df[
                    "Cluster"
                ].unique()
            ):

                cluster_data = (
                    pca_df[
                        pca_df[
                            "Cluster"
                        ]
                        ==
                        cluster
                    ]
                )

                plt.scatter(
                    cluster_data["PC1"],
                    cluster_data["PC2"],
                    label=
                        f"Cluster {cluster}",
                    alpha=0.7
                )

            plt.title(
                "Customer Segments - PCA Visualization"
            )

            plt.xlabel(
                "Principal Component 1"
            )

            plt.ylabel(
                "Principal Component 2"
            )

            plt.legend()

            plt.tight_layout()

            output_path = os.path.join(
                self.output_directory,
                "customer_segments_pca.png"
            )

            plt.savefig(
                output_path,
                bbox_inches="tight"
            )

            plt.close()

            logger.info(
                f"PCA plot saved to: "
                f"{output_path}"
            )

            logger.info(
                f"PCA explained variance ratio: "
                f"{pca.explained_variance_ratio_}"
            )

            return output_path

        except Exception as e:

            logger.error(
                "PCA customer segmentation plot failed."
            )

            raise CustomException(
                e,
                sys
            )