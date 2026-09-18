import json
import os
import sys

import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class FinalSegmentationReport:

    def generate_report(
        self,
        segmented_df: pd.DataFrame,
        segment_summary: pd.DataFrame,
        silhouette_score_value: float,
        inertia: float,
        selected_k: int,
        feature_names: list,
        preprocessor_path: str,
        model_path: str,
        visualization_paths: dict,
        output_path: str
    ) -> dict:

        try:

            logger.info(
                "Starting final customer segmentation report generation."
            )

            # ---------------------------------
            # Validate data
            # ---------------------------------

            if segmented_df.empty:

                raise ValueError(
                    "Segmented customer dataset is empty."
                )

            if segment_summary.empty:

                raise ValueError(
                    "Segment summary is empty."
                )

            if "Cluster" not in segmented_df.columns:

                raise ValueError(
                    "Cluster column not found in segmented dataset."
                )

            # ---------------------------------
            # Cluster distribution
            # ---------------------------------

            cluster_counts = (
                segmented_df[
                    "Cluster"
                ]
                .value_counts()
                .sort_index()
            )

            total_customers = (
                segmented_df[
                    "CustomerID"
                ]
                .nunique()
            )

            cluster_distribution = {}

            for cluster, count in (
                cluster_counts.items()
            ):

                cluster_distribution[
                    str(int(cluster))
                ] = {
                    "customer_count":
                        int(count),

                    "customer_percentage":
                        float(
                            (
                                count
                                /
                                total_customers
                            )
                            * 100
                        )
                }

            # ---------------------------------
            # Segment summaries
            # ---------------------------------

            segments = []

            for _, row in (
                segment_summary.iterrows()
            ):

                segment_info = {

                    "cluster":
                        int(
                            row["Cluster"]
                        ),

                    "segment_name":
                        str(
                            row[
                                "SegmentName"
                            ]
                        ),

                    "customer_count":
                        int(
                            row[
                                "CustomerCount"
                            ]
                        ),

                    "customer_percentage":
                        float(
                            row[
                                "CustomerPercentage"
                            ]
                        ),

                    "average_features": {

                        "Recency":
                            float(
                                row["Recency"]
                            ),

                        "Frequency":
                            float(
                                row["Frequency"]
                            ),

                        "Monetary":
                            float(
                                row["Monetary"]
                            ),

                        "TotalItems":
                            float(
                                row["TotalItems"]
                            ),

                        "AverageOrderValue":
                            float(
                                row[
                                    "AverageOrderValue"
                                ]
                            ),

                        "Tenure":
                            float(
                                row["Tenure"]
                            )
                    },

                    "behavioral_flags": {

                        "RecentPurchase":
                            bool(
                                row[
                                    "RecentPurchase"
                                ]
                            ),

                        "HighFrequency":
                            bool(
                                row[
                                    "HighFrequency"
                                ]
                            ),

                        "HighMonetary":
                            bool(
                                row[
                                    "HighMonetary"
                                ]
                            ),

                        "HighVolume":
                            bool(
                                row[
                                    "HighVolume"
                                ]
                            ),

                        "HighAOV":
                            bool(
                                row[
                                    "HighAOV"
                                ]
                            ),

                        "LongTenure":
                            bool(
                                row[
                                    "LongTenure"
                                ]
                            )
                    },

                    "recommendation":
                        str(
                            row[
                                "Recommendation"
                            ]
                        )
                }

                segments.append(
                    segment_info
                )

            # ---------------------------------
            # Build final report
            # ---------------------------------

            report = {

                "model": {

                    "algorithm":
                        "KMeans",

                    "selected_clusters":
                        int(
                            selected_k
                        ),

                    "random_state":
                        42,

                    "n_init":
                        10,

                    "model_path":
                        model_path,

                    "preprocessor_path":
                        preprocessor_path
                },

                "features":
                    feature_names,

                "preprocessing": {

                    "log_transformed_features": [
                        "Frequency",
                        "Monetary",
                        "TotalItems",
                        "AverageOrderValue"
                    ],

                    "non_log_features": [
                        "Recency",
                        "Tenure"
                    ],

                    "scaling":
                        "StandardScaler",

                    "customer_id_used_as_feature":
                        False
                },

                "dataset": {

                    "total_customers":
                        int(
                            total_customers
                        ),

                    "feature_count":
                        int(
                            len(
                                feature_names
                            )
                        ),

                    "cluster_distribution":
                        cluster_distribution
                },

                "evaluation": {

                    "silhouette_score":
                        float(
                            silhouette_score_value
                        ),

                    "inertia":
                        float(
                            inertia
                        )
                },

                "segments":
                    segments,

                "visualizations":
                    visualization_paths
            }

            # ---------------------------------
            # Create output directory
            # ---------------------------------

            output_directory = (
                os.path.dirname(
                    output_path
                )
            )

            if output_directory:

                os.makedirs(
                    output_directory,
                    exist_ok=True
                )

            # ---------------------------------
            # Save JSON
            # ---------------------------------

            with open(
                output_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    report,
                    file,
                    indent=4
                )

            logger.info(
                f"Final segmentation report saved to: "
                f"{output_path}"
            )

            return report

        except Exception as e:

            logger.error(
                "Final segmentation report generation failed."
            )

            raise CustomException(
                e,
                sys
            )