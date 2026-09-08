import sys
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class LogisticChurnModel:
    def build_model(
        self
    ) -> Pipeline:

        return Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42
                    )
                )
            ]
        )

    def train(
        self,
        X_train,
        y_train
    ) -> Pipeline:

        try:

            logger.info(
                "Starting logistic regression training."
            )

            model = self.build_model()

            model.fit(
                X_train,
                y_train
            )

            logger.info(
                "Logistic regression training completed."
            )

            return model

        except Exception as e:

            logger.error(
                "Logistic regression training failed."
            )

            raise CustomException(
                e,
                sys
            )