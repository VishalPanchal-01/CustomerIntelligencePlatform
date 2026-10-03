import sys

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.exception import CustomException
from src.utils.logger import logger


class CLVLinearRegressionModel:

    def build_model(
        self
    ) -> Pipeline:

        try:

            logger.info(
                "Building CLV Linear Regression pipeline."
            )

            model = Pipeline(
                steps=[
                    (
                        "scaler",
                        StandardScaler()
                    ),
                    (
                        "regressor",
                        LinearRegression()
                    )
                ]
            )

            return model

        except Exception as e:

            logger.error(
                "CLV Linear Regression model creation failed."
            )

            raise CustomException(
                e,
                sys
            )

    def train(
        self,
        X_train,
        y_train
    ) -> Pipeline:

        try:

            logger.info(
                "Starting CLV Linear Regression training."
            )

            model = (
                self.build_model()
            )

            model.fit(
                X_train,
                y_train
            )

            logger.info(
                "CLV Linear Regression training completed."
            )

            return model

        except Exception as e:

            logger.error(
                "CLV Linear Regression training failed."
            )

            raise CustomException(
                e,
                sys
            )