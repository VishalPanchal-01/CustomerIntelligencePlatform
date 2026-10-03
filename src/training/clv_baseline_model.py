import sys

from sklearn.dummy import DummyRegressor

from src.utils.exception import CustomException
from src.utils.logger import logger


class CLVBaselineModel:

    def __init__(
        self,
        strategy: str = "median"
    ):

        self.strategy = strategy

    def build_model(
        self
    ) -> DummyRegressor:

        try:

            logger.info(
                f"Building CLV baseline model "
                f"with strategy={self.strategy}."
            )

            model = DummyRegressor(
                strategy=self.strategy
            )

            return model

        except Exception as e:

            logger.error(
                "CLV baseline model creation failed."
            )

            raise CustomException(
                e,
                sys
            )

    def train(
        self,
        X_train,
        y_train
    ) -> DummyRegressor:

        try:

            logger.info(
                "Starting CLV baseline model training."
            )

            model = self.build_model()

            model.fit(
                X_train,
                y_train
            )

            logger.info(
                "CLV baseline model training completed."
            )

            return model

        except Exception as e:

            logger.error(
                "CLV baseline model training failed."
            )

            raise CustomException(
                e,
                sys
            )