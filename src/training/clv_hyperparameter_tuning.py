import sys

import numpy as np

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV
)

from src.utils.exception import CustomException
from src.utils.logger import logger


class CLVHyperparameterTuner:

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42
    ):

        self.n_splits = n_splits
        self.random_state = random_state

        self.cv = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

    # ==================================================
    # RANDOM FOREST - RAW TARGET
    # ==================================================

    def tune_random_forest_raw(
        self,
        X_train,
        y_train,
        n_iter: int = 15
    ) -> RandomizedSearchCV:

        try:

            logger.info(
                "Starting Random Forest RAW target tuning."
            )

            model = RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=1
            )

            parameter_distribution = {

                "n_estimators": [
                    100,
                    200,
                    300,
                    500
                ],

                "max_depth": [
                    None,
                    5,
                    10,
                    15,
                    20
                ],

                "min_samples_split": [
                    2,
                    5,
                    10
                ],

                "min_samples_leaf": [
                    1,
                    2,
                    4
                ],

                "max_features": [
                    "sqrt",
                    "log2",
                    1.0
                ]
            }

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=
                    parameter_distribution,
                n_iter=n_iter,
                scoring=
                    "neg_mean_absolute_error",
                cv=self.cv,
                random_state=
                    self.random_state,
                n_jobs=-1,
                verbose=1,
                refit=True
            )

            search.fit(
                X_train,
                y_train
            )

            logger.info(
                "Random Forest RAW tuning completed."
            )

            logger.info(
                f"Best parameters: "
                f"{search.best_params_}"
            )

            logger.info(
                f"Best CV MAE: "
                f"{-search.best_score_}"
            )

            return search

        except Exception as e:

            logger.error(
                "Random Forest RAW tuning failed."
            )

            raise CustomException(
                e,
                sys
            )

    # ==================================================
    # RANDOM FOREST - LOG TARGET
    # ==================================================

    def tune_random_forest_log(
        self,
        X_train,
        y_train_raw,
        n_iter: int = 15
    ) -> RandomizedSearchCV:

        try:

            logger.info(
                "Starting Random Forest LOG target tuning."
            )

            base_model = RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=1
            )

            model = TransformedTargetRegressor(
                regressor=base_model,
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=False
            )

            parameter_distribution = {

                "regressor__n_estimators": [
                    100,
                    200,
                    300,
                    500
                ],

                "regressor__max_depth": [
                    None,
                    5,
                    10,
                    15,
                    20
                ],

                "regressor__min_samples_split": [
                    2,
                    5,
                    10
                ],

                "regressor__min_samples_leaf": [
                    1,
                    2,
                    4
                ],

                "regressor__max_features": [
                    "sqrt",
                    "log2",
                    1.0
                ]
            }

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=
                    parameter_distribution,
                n_iter=n_iter,
                scoring=
                    "neg_mean_absolute_error",
                cv=self.cv,
                random_state=
                    self.random_state,
                n_jobs=-1,
                verbose=1,
                refit=True
            )

            search.fit(
                X_train,
                y_train_raw
            )

            logger.info(
                "Random Forest LOG tuning completed."
            )

            logger.info(
                f"Best parameters: "
                f"{search.best_params_}"
            )

            logger.info(
                f"Best CV MAE "
                f"(original revenue scale): "
                f"{-search.best_score_}"
            )

            return search

        except Exception as e:

            logger.error(
                "Random Forest LOG tuning failed."
            )

            raise CustomException(
                e,
                sys
            )

    # ==================================================
    # GRADIENT BOOSTING - RAW TARGET
    # ==================================================

    def tune_gradient_boosting_raw(
        self,
        X_train,
        y_train,
        n_iter: int = 15
    ) -> RandomizedSearchCV:

        try:

            logger.info(
                "Starting Gradient Boosting "
                "RAW target tuning."
            )

            model = GradientBoostingRegressor(
                random_state=self.random_state
            )

            parameter_distribution = {

                "n_estimators": [
                    100,
                    200,
                    300,
                    500
                ],

                "learning_rate": [
                    0.01,
                    0.03,
                    0.05,
                    0.10
                ],

                "max_depth": [
                    2,
                    3,
                    4,
                    5
                ],

                "min_samples_split": [
                    2,
                    5,
                    10
                ],

                "min_samples_leaf": [
                    1,
                    2,
                    4
                ],

                "subsample": [
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ],

                "max_features": [
                    None,
                    "sqrt",
                    "log2"
                ]
            }

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=
                    parameter_distribution,
                n_iter=n_iter,
                scoring=
                    "neg_mean_absolute_error",
                cv=self.cv,
                random_state=
                    self.random_state,
                n_jobs=-1,
                verbose=1,
                refit=True
            )

            search.fit(
                X_train,
                y_train
            )

            logger.info(
                "Gradient Boosting RAW "
                "tuning completed."
            )

            logger.info(
                f"Best parameters: "
                f"{search.best_params_}"
            )

            logger.info(
                f"Best CV MAE: "
                f"{-search.best_score_}"
            )

            return search

        except Exception as e:

            logger.error(
                "Gradient Boosting RAW tuning failed."
            )

            raise CustomException(
                e,
                sys
            )

    # ==================================================
    # GRADIENT BOOSTING - LOG TARGET
    # ==================================================

    def tune_gradient_boosting_log(
        self,
        X_train,
        y_train_raw,
        n_iter: int = 15
    ) -> RandomizedSearchCV:

        try:

            logger.info(
                "Starting Gradient Boosting "
                "LOG target tuning."
            )

            base_model = GradientBoostingRegressor(
                random_state=self.random_state
            )

            model = TransformedTargetRegressor(
                regressor=base_model,
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=False
            )

            parameter_distribution = {

                "regressor__n_estimators": [
                    100,
                    200,
                    300,
                    500
                ],

                "regressor__learning_rate": [
                    0.01,
                    0.03,
                    0.05,
                    0.10
                ],

                "regressor__max_depth": [
                    2,
                    3,
                    4,
                    5
                ],

                "regressor__min_samples_split": [
                    2,
                    5,
                    10
                ],

                "regressor__min_samples_leaf": [
                    1,
                    2,
                    4
                ],

                "regressor__subsample": [
                    0.7,
                    0.8,
                    0.9,
                    1.0
                ],

                "regressor__max_features": [
                    None,
                    "sqrt",
                    "log2"
                ]
            }

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=
                    parameter_distribution,
                n_iter=n_iter,
                scoring=
                    "neg_mean_absolute_error",
                cv=self.cv,
                random_state=
                    self.random_state,
                n_jobs=-1,
                verbose=1,
                refit=True
            )

            search.fit(
                X_train,
                y_train_raw
            )

            logger.info(
                "Gradient Boosting LOG "
                "tuning completed."
            )

            logger.info(
                f"Best parameters: "
                f"{search.best_params_}"
            )

            logger.info(
                f"Best CV MAE "
                f"(original revenue scale): "
                f"{-search.best_score_}"
            )

            return search

        except Exception as e:

            logger.error(
                "Gradient Boosting LOG tuning failed."
            )

            raise CustomException(
                e,
                sys
            )