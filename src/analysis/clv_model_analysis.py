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
                "Analyzing CLV Random Forest "
                "feature importance."
            )

            estimator = (
                self._get_underlying_regressor(
                    model
                )
            )

            return (
                self._build_feature_importance_report(
                    estimator,
                    feature_names
                )
            )

        except Exception as e:

            logger.error(
                "CLV Random Forest importance "
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

            estimator = (
                self._get_underlying_regressor(
                    model
                )
            )

            return (
                self._build_feature_importance_report(
                    estimator,
                    feature_names
                )
            )

        except Exception as e:

            logger.error(
                "CLV Gradient Boosting importance "
                "analysis failed."
            )

            raise CustomException(
                e,
                sys
            )

    def analyze_tree_importance(
        self,
        model,
        feature_names: list
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Analyzing final CLV tree-model "
                "feature importance."
            )

            estimator = (
                self._get_underlying_regressor(
                    model
                )
            )

            if not hasattr(
                estimator,
                "feature_importances_"
            ):

                raise ValueError(
                    "Selected CLV model does not expose "
                    "feature_importances_."
                )

            return (
                self._build_feature_importance_report(
                    estimator,
                    feature_names
                )
            )

        except Exception as e:

            logger.error(
                "Final CLV feature importance "
                "analysis failed."
            )

            raise CustomException(
                e,
                sys
            )

    def _get_underlying_regressor(
        self,
        model
    ):

        # ---------------------------------
        # TransformedTargetRegressor
        # ---------------------------------

        if hasattr(
            model,
            "regressor_"
        ):

            return model.regressor_

        # ---------------------------------
        # Normal estimator
        # ---------------------------------

        return model

    def _build_feature_importance_report(
        self,
        model,
        feature_names: list
    ) -> pd.DataFrame:

        if not hasattr(
            model,
            "feature_importances_"
        ):

            raise ValueError(
                "Model does not contain "
                "feature_importances_."
            )

        importances = (
            model.feature_importances_
        )

        if len(importances) != len(
            feature_names
        ):

            raise ValueError(
                "Feature-name count does not match "
                "feature importance count."
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