import os
import sys

import joblib

from src.utils.exception import CustomException
from src.utils.logger import logger


class CLVModelPersistence:

    @staticmethod
    def save_model(
        model,
        file_path: str
    ) -> None:

        try:

            logger.info(
                f"Saving CLV model to: "
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
                "CLV model saved successfully."
            )

        except Exception as e:

            logger.error(
                "CLV model saving failed."
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
                f"Loading CLV model from: "
                f"{file_path}"
            )

            if not os.path.exists(
                file_path
            ):

                raise FileNotFoundError(
                    f"CLV model not found: "
                    f"{file_path}"
                )

            model = joblib.load(
                file_path
            )

            logger.info(
                "CLV model loaded successfully."
            )

            return model

        except Exception as e:

            logger.error(
                "CLV model loading failed."
            )

            raise CustomException(
                e,
                sys
            )