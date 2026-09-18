import json
import os
import sys

import numpy as np
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class FinalChurnReport:

    def generate_report(
        self,
        model_name: str,
        model,
        evaluation_results: dict,
        churn_df: pd.DataFrame,
        feature_names: list,
        model_path: str,
        prediction_days: int,
        output_path: str
    ) -> dict:

        try:

            logger.info(
                "Starting final churn model report generation."
            )

            # ---------------------------------
            # Validate churn dataset
            # ---------------------------------

            if churn_df.empty:

                raise ValueError(
                    "Churn dataset is empty."
                )

            if "Churn" not in churn_df.columns:

                raise ValueError(
                    "Churn column not found in dataset."
                )

            # ---------------------------------
            # Churn distribution
            # ---------------------------------

            churn_counts = (
                churn_df["Churn"]
                .value_counts()
                .sort_index()
            )

            churn_percentages = (
                churn_df["Churn"]
                .value_counts(
                    normalize=True
                )
                .sort_index()
                * 100
            )

            churn_distribution = {}

            for churn_class in [0, 1]:

                churn_distribution[
                    str(churn_class)
                ] = {
                    "count": int(
                        churn_counts.get(
                            churn_class,
                            0
                        )
                    ),

                    "percentage": float(
                        churn_percentages.get(
                            churn_class,
                            0
                        )
                    )
                }

            # ---------------------------------
            # Confusion matrix
            # ---------------------------------

            confusion_matrix = (
                evaluation_results[
                    "confusion_matrix"
                ]
            )

            confusion_matrix_list = (
                np.asarray(
                    confusion_matrix
                )
                .tolist()
            )

            # ---------------------------------
            # Model parameters
            # ---------------------------------

            if hasattr(
                model,
                "get_params"
            ):

                model_parameters = (
                    model.get_params(
                        deep=True
                    )
                )

            else:

                model_parameters = {}

            # Convert model parameters
            # into JSON-compatible values
            clean_parameters = {}

            for key, value in (
                model_parameters.items()
            ):

                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool
                    )
                ) or value is None:

                    clean_parameters[
                        key
                    ] = value

                else:

                    clean_parameters[
                        key
                    ] = str(
                        value
                    )

            # ---------------------------------
            # Build report
            # ---------------------------------

            report = {

                "model": {
                    "name":
                        model_name,

                    "artifact_path":
                        model_path,

                    "parameters":
                        clean_parameters
                },

                "churn_definition": {
                    "positive_class":
                        1,

                    "negative_class":
                        0,

                    "prediction_window_days":
                        prediction_days,

                    "definition":
                        (
                            "Churn = 1 when an observation-period "
                            "customer makes no valid purchase during "
                            "the prediction window. Churn = 0 when "
                            "the customer purchases during the "
                            "prediction window."
                        )
                },

                "features":
                    feature_names,

                "dataset": {
                    "customers":
                        int(
                            len(
                                churn_df
                            )
                        ),

                    "feature_count":
                        int(
                            len(
                                feature_names
                            )
                        ),

                    "churn_distribution":
                        churn_distribution
                },

                "test_metrics": {

                    "accuracy":
                        float(
                            evaluation_results[
                                "accuracy"
                            ]
                        ),

                    "precision":
                        float(
                            evaluation_results[
                                "precision"
                            ]
                        ),

                    "recall":
                        float(
                            evaluation_results[
                                "recall"
                            ]
                        ),

                    "f1_score":
                        float(
                            evaluation_results[
                                "f1_score"
                            ]
                        ),

                    "roc_auc":
                        (
                            float(
                                evaluation_results[
                                    "roc_auc"
                                ]
                            )
                            if evaluation_results[
                                "roc_auc"
                            ] is not None
                            else None
                        ),

                    "confusion_matrix":
                        confusion_matrix_list
                }
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
            # Save JSON report
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
                f"Final churn report saved to: "
                f"{output_path}"
            )

            return report

        except Exception as e:

            logger.error(
                "Final churn model report generation failed."
            )

            raise CustomException(
                e,
                sys
            )