import sys

import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger
from src.utils.segmentation_persistence import (
    SegmentationPersistence
)


class SegmentPredictor:

    def __init__(
        self,
        preprocessor_path: str = (
            "models/segmentation/"
            "segmentation_preprocessor.pkl"
        ),
        model_path: str = (
            "models/segmentation/"
            "kmeans_model.pkl"
        ),
        mapping_path: str = (
            "artifacts/segmentation/"
            "segment_summary.csv"
        )
    ):

        try:

            logger.info(
                "Initializing customer segment predictor."
            )

            self.features = [
                "Recency",
                "Frequency",
                "Monetary",
                "TotalItems",
                "AverageOrderValue",
                "Tenure"
            ]

            persistence = (
                SegmentationPersistence()
            )

            # ---------------------------------
            # Load fitted preprocessor
            # ---------------------------------

            self.preprocessor = (
                persistence.load_artifact(
                    preprocessor_path
                )
            )

            # ---------------------------------
            # Load trained K-Means
            # ---------------------------------

            self.model = (
                persistence.load_artifact(
                    model_path
                )
            )

            # ---------------------------------
            # Load cluster-to-segment mapping
            # ---------------------------------

            mapping_df = pd.read_csv(
                mapping_path
            )

            required_mapping_columns = [
                "Cluster",
                "SegmentName",
                "Recommendation"
            ]

            missing_mapping_columns = [
                column
                for column
                in required_mapping_columns
                if column not in mapping_df.columns
            ]

            if missing_mapping_columns:

                raise ValueError(
                    f"Missing segment mapping columns: "
                    f"{missing_mapping_columns}"
                )

            self.segment_mapping = (
                mapping_df[
                    [
                        "Cluster",
                        "SegmentName"
                    ]
                ]
                .set_index(
                    "Cluster"
                )[
                    "SegmentName"
                ]
                .to_dict()
            )

            self.recommendation_mapping = (
                mapping_df[
                    [
                        "Cluster",
                        "Recommendation"
                    ]
                ]
                .set_index(
                    "Cluster"
                )[
                    "Recommendation"
                ]
                .to_dict()
            )

            logger.info(
                "Customer segment predictor initialized."
            )

        except Exception as e:

            logger.error(
                "Segment predictor initialization failed."
            )

            raise CustomException(
                e,
                sys
            )

    def predict_segment(
        self,
        customer_data: dict
    ) -> dict:

        try:

            logger.info(
                "Starting customer segment prediction."
            )

            # ---------------------------------
            # Validate required features
            # ---------------------------------

            missing_features = [
                feature
                for feature in self.features
                if feature not in customer_data
            ]

            if missing_features:

                raise ValueError(
                    f"Missing required features: "
                    f"{missing_features}"
                )

            # ---------------------------------
            # Create one-row dataframe
            # ---------------------------------

            input_df = pd.DataFrame(
                [
                    {
                        feature:
                            customer_data[
                                feature
                            ]
                        for feature
                        in self.features
                    }
                ]
            )

            # ---------------------------------
            # Validate missing values
            # ---------------------------------

            if (
                input_df
                .isnull()
                .any()
                .any()
            ):

                raise ValueError(
                    "Customer segmentation features "
                    "contain missing values."
                )

            # ---------------------------------
            # Validate numeric values
            # ---------------------------------

            for feature in self.features:

                if not pd.api.types.is_numeric_dtype(
                    input_df[
                        feature
                    ]
                ):

                    raise ValueError(
                        f"{feature} must be numeric."
                    )

            # ---------------------------------
            # Validate non-negative values
            # ---------------------------------

            if (
                input_df[
                    self.features
                ]
                < 0
            ).any().any():

                raise ValueError(
                    "Customer segmentation features "
                    "cannot be negative."
                )

            # ---------------------------------
            # Apply fitted preprocessing
            # ---------------------------------

            scaled_features = (
                self.preprocessor
                .transform_features(
                    input_df
                )
            )

            # ---------------------------------
            # Predict cluster
            # ---------------------------------

            cluster = int(
                self.model.predict(
                    scaled_features
                )[0]
            )

            # ---------------------------------
            # Map cluster to business segment
            # ---------------------------------

            segment_name = (
                self.segment_mapping.get(
                    cluster,
                    "Unknown Segment"
                )
            )

            recommendation = (
                self.recommendation_mapping.get(
                    cluster,
                    (
                        "Monitor customer behavior "
                        "and apply personalized engagement."
                    )
                )
            )

            # ---------------------------------
            # Build response
            # ---------------------------------

            result = {
                "cluster":
                    cluster,

                "segment_name":
                    segment_name,

                "recommendation":
                    recommendation
            }

            logger.info(
                f"Customer segment prediction: "
                f"{result}"
            )

            return result

        except Exception as e:

            logger.error(
                "Customer segment prediction failed."
            )

            raise CustomException(
                e,
                sys
            )