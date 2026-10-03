import sys

import pandas as pd

from src.training.clv_baseline_model import (
    CLVBaselineModel
)

from src.training.clv_linear_model import (
    CLVLinearRegressionModel
)

from src.training.clv_random_forest_model import (
    CLVRandomForestModel
)

from src.training.clv_gradient_boosting_model import (
    CLVGradientBoostingModel
)

from src.evaluation.clv_evaluation import (
    CLVModelEvaluation
)

from src.utils.exception import CustomException
from src.utils.logger import logger


class CLVModelComparison:

    def compare_models(
        self,
        X_train,
        X_test,
        y_raw_train,
        y_raw_test,
        y_log_train
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Starting CLV model comparison."
            )

            evaluator = (
                CLVModelEvaluation()
            )

            results = []

            # =================================
            # DUMMY REGRESSOR
            # =================================

            logger.info(
                "Training Dummy Regressor."
            )

            dummy_raw = (
                CLVBaselineModel(
                    strategy="median"
                )
                .train(
                    X_train,
                    y_raw_train
                )
            )

            dummy_raw_results = (
                evaluator.evaluate(
                    model=dummy_raw,
                    X_test=X_test,
                    y_test_raw=y_raw_test,
                    target_type="raw"
                )
            )

            results.append(
                self._create_result_row(
                    model_name=
                        "Dummy Regressor",
                    target_type=
                        "Raw",
                    evaluation=
                        dummy_raw_results
                )
            )

            dummy_log = (
                CLVBaselineModel(
                    strategy="median"
                )
                .train(
                    X_train,
                    y_log_train
                )
            )

            dummy_log_results = (
                evaluator.evaluate(
                    model=dummy_log,
                    X_test=X_test,
                    y_test_raw=y_raw_test,
                    target_type="log"
                )
            )

            results.append(
                self._create_result_row(
                    model_name=
                        "Dummy Regressor",
                    target_type=
                        "Log",
                    evaluation=
                        dummy_log_results
                )
            )

            # =================================
            # LINEAR REGRESSION
            # =================================

            logger.info(
                "Training Linear Regression."
            )

            linear_raw = (
                CLVLinearRegressionModel()
                .train(
                    X_train,
                    y_raw_train
                )
            )

            linear_raw_results = (
                evaluator.evaluate(
                    model=linear_raw,
                    X_test=X_test,
                    y_test_raw=y_raw_test,
                    target_type="raw"
                )
            )

            results.append(
                self._create_result_row(
                    model_name=
                        "Linear Regression",
                    target_type=
                        "Raw",
                    evaluation=
                        linear_raw_results
                )
            )

            linear_log = (
                CLVLinearRegressionModel()
                .train(
                    X_train,
                    y_log_train
                )
            )

            linear_log_results = (
                evaluator.evaluate(
                    model=linear_log,
                    X_test=X_test,
                    y_test_raw=y_raw_test,
                    target_type="log"
                )
            )

            results.append(
                self._create_result_row(
                    model_name=
                        "Linear Regression",
                    target_type=
                        "Log",
                    evaluation=
                        linear_log_results
                )
            )

            # =================================
            # RANDOM FOREST
            # =================================

            logger.info(
                "Training Random Forest."
            )

            rf_raw = (
                CLVRandomForestModel()
                .train(
                    X_train,
                    y_raw_train
                )
            )

            rf_raw_results = (
                evaluator.evaluate(
                    model=rf_raw,
                    X_test=X_test,
                    y_test_raw=y_raw_test,
                    target_type="raw"
                )
            )

            results.append(
                self._create_result_row(
                    model_name=
                        "Random Forest",
                    target_type=
                        "Raw",
                    evaluation=
                        rf_raw_results
                )
            )

            rf_log = (
                CLVRandomForestModel()
                .train(
                    X_train,
                    y_log_train
                )
            )

            rf_log_results = (
                evaluator.evaluate(
                    model=rf_log,
                    X_test=X_test,
                    y_test_raw=y_raw_test,
                    target_type="log"
                )
            )

            results.append(
                self._create_result_row(
                    model_name=
                        "Random Forest",
                    target_type=
                        "Log",
                    evaluation=
                        rf_log_results
                )
            )

            # =================================
            # GRADIENT BOOSTING
            # =================================

            logger.info(
                "Training Gradient Boosting."
            )

            gb_raw = (
                CLVGradientBoostingModel()
                .train(
                    X_train,
                    y_raw_train
                )
            )

            gb_raw_results = (
                evaluator.evaluate(
                    model=gb_raw,
                    X_test=X_test,
                    y_test_raw=y_raw_test,
                    target_type="raw"
                )
            )

            results.append(
                self._create_result_row(
                    model_name=
                        "Gradient Boosting",
                    target_type=
                        "Raw",
                    evaluation=
                        gb_raw_results
                )
            )

            gb_log = (
                CLVGradientBoostingModel()
                .train(
                    X_train,
                    y_log_train
                )
            )

            gb_log_results = (
                evaluator.evaluate(
                    model=gb_log,
                    X_test=X_test,
                    y_test_raw=y_raw_test,
                    target_type="log"
                )
            )

            results.append(
                self._create_result_row(
                    model_name=
                        "Gradient Boosting",
                    target_type=
                        "Log",
                    evaluation=
                        gb_log_results
                )
            )

            # =================================
            # BUILD REPORT
            # =================================

            comparison = pd.DataFrame(
                results
            )

            comparison = (
                comparison
                .sort_values(
                    by=[
                        "MAE",
                        "RMSE"
                    ],
                    ascending=[
                        True,
                        True
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            logger.info(
                f"CLV model comparison completed:\n"
                f"{comparison}"
            )

            return comparison

        except Exception as e:

            logger.error(
                "CLV model comparison failed."
            )

            raise CustomException(
                e,
                sys
            )

    def _create_result_row(
        self,
        model_name: str,
        target_type: str,
        evaluation: dict
    ) -> dict:

        return {

            "Model":
                model_name,

            "Target":
                target_type,

            "MAE":
                evaluation[
                    "mae"
                ],

            "RMSE":
                evaluation[
                    "rmse"
                ],

            "R2":
                evaluation[
                    "r2"
                ],

            "NegativePredictions":
                evaluation[
                    "negative_prediction_count"
                ],

            "NegativePredictionPercentage":
                evaluation[
                    "negative_prediction_percentage"
                ]
        }