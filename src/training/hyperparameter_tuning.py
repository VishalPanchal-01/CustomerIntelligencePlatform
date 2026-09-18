import sys

import pandas as pd

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold
)

from src.training.logistic_model import (
    LogisticChurnModel
)

from src.training.random_forest_model import (
    RandomForestChurnModel
)

from src.utils.exception import CustomException
from src.utils.logger import logger


class ChurnHyperparameterTuner:

    def __init__(self):

        self.cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

    # Logistic Regression
    def tune_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ):

        try:

            logger.info(
                "Starting Logistic Regression tuning."
            )

            trainer = (
                LogisticChurnModel()
            )

            model = (
                trainer.build_model()
            )

            parameter_grid = {

                "classifier__C": [
                    0.01,
                    0.1,
                    1,
                    10,
                    100
                ],

                "classifier__class_weight": [
                    None,
                    "balanced"
                ],

                "classifier__solver": [
                    "liblinear",
                    "lbfgs"
                ]
            }

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=parameter_grid,
                n_iter=10,
                scoring="f1",
                cv=self.cv,
                random_state=42,
                n_jobs=-1,
                refit=True
            )

            search.fit(
                X_train,
                y_train
            )

            logger.info(
                f"Best Logistic Regression parameters: "
                f"{search.best_params_}"
            )

            logger.info(
                f"Best Logistic Regression CV F1: "
                f"{search.best_score_}"
            )

            return search

        except Exception as e:

            logger.error(
                "Logistic Regression tuning failed."
            )

            raise CustomException(
                e,
                sys
            )

    # Random Forest:
    def tune_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ):

        try:

            logger.info(
                "Starting Random Forest tuning."
            )

            trainer = (
                RandomForestChurnModel()
            )

            model = (
                trainer.build_model()
            )

            parameter_grid = {

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
                    None
                ],

                "class_weight": [
                    None,
                    "balanced"
                ]
            }

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=parameter_grid,
                n_iter=30,
                scoring="f1",
                cv=self.cv,
                random_state=42,
                n_jobs=-1,
                refit=True,
                verbose=1
            )

            search.fit(
                X_train,
                y_train
            )

            logger.info(
                f"Best Random Forest parameters: "
                f"{search.best_params_}"
            )

            logger.info(
                f"Best Random Forest CV F1: "
                f"{search.best_score_}"
            )

            return search

        except Exception as e:

            logger.error(
                "Random Forest tuning failed."
            )

            raise CustomException(
                e,
                sys
            )