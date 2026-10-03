import os
import sys

import joblib

from src.utils.exception import CustomException
from src.utils.logger import logger


class RecommendationPersistence:

    @staticmethod
    def save_model(
        model,
        file_path: str
    ) -> None:

        try:

            logger.info(
                f"Saving recommendation model to: "
                f"{file_path}"
            )

            directory = os.path.dirname(
                file_path
            )

            if directory:

                os.makedirs(
                    directory,
                    exist_ok=True
                )

            joblib.dump(
                model,
                file_path
            )

            logger.info(
                "Recommendation model saved successfully."
            )

        except Exception as e:

            logger.error(
                "Recommendation model saving failed."
            )

            raise CustomException(
                e,
                sys
            )

    @staticmethod
    def load_model(
        file_path: str
    ):

        try:

            logger.info(
                f"Loading recommendation model from: "
                f"{file_path}"
            )

            if not os.path.exists(
                file_path
            ):

                raise FileNotFoundError(
                    f"Recommendation model "
                    f"not found: {file_path}"
                )

            model = joblib.load(
                file_path
            )

            logger.info(
                "Recommendation model loaded successfully."
            )

            return model

        except Exception as e:

            logger.error(
                "Recommendation model loading failed."
            )

            raise CustomException(
                e,
                sys
            )