import sys

import pandas as pd

from src.evaluation.churn_evaluation import (
    ChurnModelEvaluation
)

from src.utils.exception import CustomException
from src.utils.logger import logger


class FinalModelSelector:

    def select_best_model(
        self,
        models: dict,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ):

        try:

            logger.info(
                "Starting final model selection."
            )

            evaluator = (
                ChurnModelEvaluation()
            )

            results = []

            for model_name, model in models.items():

                logger.info(
                    f"Evaluating final model: "
                    f"{model_name}"
                )

                evaluation = (
                    evaluator.evaluate(
                        model,
                        X_test,
                        y_test
                    )
                )

                results.append(
                    {
                        "Model":
                            model_name,

                        "Accuracy":
                            evaluation[
                                "accuracy"
                            ],

                        "Precision":
                            evaluation[
                                "precision"
                            ],

                        "Recall":
                            evaluation[
                                "recall"
                            ],

                        "F1Score":
                            evaluation[
                                "f1_score"
                            ],

                        "ROCAUC":
                            evaluation[
                                "roc_auc"
                            ],

                        "ConfusionMatrix":
                            evaluation[
                                "confusion_matrix"
                            ],

                        "ModelObject":
                            model
                    }
                )

            comparison = (
                pd.DataFrame(
                    results
                )
            )

            comparison = (
                comparison
                .sort_values(
                    by=[
                        "F1Score",
                        "ROCAUC",
                        "Recall"
                    ],
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )

            best_model_name = (
                comparison.iloc[0][
                    "Model"
                ]
            )

            best_model = (
                comparison.iloc[0][
                    "ModelObject"
                ]
            )

            logger.info(
                f"Selected model: "
                f"{best_model_name}"
            )

            return (
                best_model_name,
                best_model,
                comparison
            )

        except Exception as e:

            logger.error(
                "Final model selection failed."
            )

            raise CustomException(
                e,
                sys
            )