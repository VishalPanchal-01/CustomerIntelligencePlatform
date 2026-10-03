import sys

import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class CLVModelAnalysis:

    def analyze_linear_coefficients(
        self,
        model,
        feature_names: list
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Analyzing CLV Linear Regression coefficients."
            )

            regressor = (
                model.named_steps[
                    "regressor"
                ]
            )

            coefficients = (
                regressor.coef_
            )

            result = pd.DataFrame(
                {
                    "Feature":
                        feature_names,

                    "Coefficient":
                        coefficients
                }
            )

            result[
                "AbsoluteCoefficient"
            ] = (
                result[
                    "Coefficient"
                ]
                .abs()
            )

            result = (
                result
                .sort_values(
                    by="AbsoluteCoefficient",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )

            logger.info(
                f"CLV Linear Regression coefficients:\n"
                f"{result}"
            )

            return result

        except Exception as e:

            logger.error(
                "CLV coefficient analysis failed."
            )

            raise CustomException(
                e,
                sys
            )

    def analyze_random_forest_importance(
        self,
        model,
        feature_names: list
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Analyzing CLV Random Forest feature importance."
            )

            result = (
                self._build_feature_importance_report(
                    model,
                    feature_names
                )
            )

            logger.info(
                f"CLV Random Forest feature importance:\n"
                f"{result}"
            )

            return result

        except Exception as e:

            logger.error(
                "CLV Random Forest feature importance "
                "analysis failed."
            )

            raise CustomException(
                e,
                sys
            )

    def analyze_gradient_boosting_importance(
        self,
        model,
        feature_names: list
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Analyzing CLV Gradient Boosting "
                "feature importance."
            )

            result = (
                self._build_feature_importance_report(
                    model,
                    feature_names
                )
            )

            logger.info(
                f"CLV Gradient Boosting "
                f"feature importance:\n"
                f"{result}"
            )

            return result

        except Exception as e:

            logger.error(
                "CLV Gradient Boosting feature "
                "importance analysis failed."
            )

            raise CustomException(
                e,
                sys
            )

    def _build_feature_importance_report(
        self,
        model,
        feature_names: list
    ) -> pd.DataFrame:

        importances = (
            model.feature_importances_
        )

        result = pd.DataFrame(
            {
                "Feature":
                    feature_names,

                "Importance":
                    importances
            }
        )

        result[
            "ImportancePercentage"
        ] = (
            result[
                "Importance"
            ]
            * 100
        )

        result = (
            result
            .sort_values(
                by="Importance",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        result[
            "Importance"
        ] = (
            result[
                "Importance"
            ]
            .round(6)
        )

        result[
            "ImportancePercentage"
        ] = (
            result[
                "ImportancePercentage"
            ]
            .round(2)
        )

        return result