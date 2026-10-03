import sys

import pandas as pd

from sklearn.model_selection import train_test_split

from src.utils.exception import CustomException
from src.utils.logger import logger


class CLVDataSplitter:

    def split_data(
        self,
        X: pd.DataFrame,
        y_raw: pd.Series,
        y_log: pd.Series,
        test_size: float = 0.20,
        random_state: int = 42
    ):

        try:

            logger.info(
                "Starting CLV train/test split."
            )

            if X.empty:

                raise ValueError(
                    "CLV feature dataset is empty."
                )

            if len(X) != len(y_raw):

                raise ValueError(
                    "X and raw target lengths do not match."
                )

            if len(X) != len(y_log):

                raise ValueError(
                    "X and log target lengths do not match."
                )

            # ---------------------------------
            # Split indices once
            # ---------------------------------

            train_indices, test_indices = (
                train_test_split(
                    X.index,
                    test_size=test_size,
                    random_state=random_state
                )
            )

            # ---------------------------------
            # Feature split
            # ---------------------------------

            X_train = (
                X.loc[
                    train_indices
                ]
                .copy()
            )

            X_test = (
                X.loc[
                    test_indices
                ]
                .copy()
            )

            # ---------------------------------
            # Raw target split
            # ---------------------------------

            y_raw_train = (
                y_raw.loc[
                    train_indices
                ]
                .copy()
            )

            y_raw_test = (
                y_raw.loc[
                    test_indices
                ]
                .copy()
            )

            # ---------------------------------
            # Log target split
            # ---------------------------------

            y_log_train = (
                y_log.loc[
                    train_indices
                ]
                .copy()
            )

            y_log_test = (
                y_log.loc[
                    test_indices
                ]
                .copy()
            )

            logger.info(
                "CLV train/test split completed."
            )

            logger.info(
                f"Training rows: "
                f"{len(X_train)}"
            )

            logger.info(
                f"Testing rows: "
                f"{len(X_test)}"
            )

            return (
                X_train,
                X_test,
                y_raw_train,
                y_raw_test,
                y_log_train,
                y_log_test
            )

        except Exception as e:

            logger.error(
                "CLV train/test split failed."
            )

            raise CustomException(
                e,
                sys
            )