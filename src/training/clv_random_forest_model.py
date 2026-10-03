import sys

from sklearn.ensemble import RandomForestRegressor

from src.utils.exception import CustomException
from src.utils.logger import logger


class CLVRandomForestModel:

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 42
    ):

        self.n_estimators = n_estimators
        self.random_state = random_state

    def build_model(
        self
    ) -> RandomForestRegressor:

        try:

            logger.info(
                "Building CLV Random Forest Regressor."
            )

            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )

            return model

        except Exception as e:

            logger.error(
                "CLV Random Forest model creation failed."
            )

            raise CustomException(
                e,
                sys
            )

    def train(
        self,
        X_train,
        y_train
    ) -> RandomForestRegressor:

        try:

            logger.info(
                "Starting CLV Random Forest training."
            )

            model = (
                self.build_model()
            )

            model.fit(
                X_train,
                y_train
            )

            logger.info(
                "CLV Random Forest training completed."
            )

            return model

        except Exception as e:

            logger.error(
                "CLV Random Forest training failed."
            )

            raise CustomException(
                e,
                sys
            )