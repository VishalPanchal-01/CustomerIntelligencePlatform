import sys
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger

from sklearn.ensemble import RandomForestClassifier

class RandomForestChurnModel:

    def build_model(
        self
    ) -> RandomForestClassifier:

        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

    def train(
        self,
        X_train,
        y_train
    ) -> RandomForestClassifier:

        try:

            logger.info(
                "Starting random forest training."
            )

            model = self.build_model()

            model.fit(
                X_train,
                y_train
            )

            logger.info(
                "Random forest training completed."
            )

            return model

        except Exception as e:

            logger.error(
                "Random forest training failed."
            )

            raise CustomException(
                e,
                sys
            )