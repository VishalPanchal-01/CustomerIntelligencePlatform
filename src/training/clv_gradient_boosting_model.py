import sys

from sklearn.ensemble import GradientBoostingRegressor

from src.utils.exception import CustomException
from src.utils.logger import logger


class CLVGradientBoostingModel:

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        random_state: int = 42
    ):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def build_model(
        self
    ) -> GradientBoostingRegressor:

        try:

            logger.info(
                "Building CLV Gradient Boosting Regressor."
            )

            model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state
            )

            return model

        except Exception as e:

            logger.error(
                "CLV Gradient Boosting model creation failed."
            )

            raise CustomException(
                e,
                sys
            )

    def train(
        self,
        X_train,
        y_train
    ) -> GradientBoostingRegressor:

        try:

            logger.info(
                "Starting CLV Gradient Boosting training."
            )

            model = (
                self.build_model()
            )

            model.fit(
                X_train,
                y_train
            )

            logger.info(
                "CLV Gradient Boosting training completed."
            )

            return model

        except Exception as e:

            logger.error(
                "CLV Gradient Boosting training failed."
            )

            raise CustomException(
                e,
                sys
            )