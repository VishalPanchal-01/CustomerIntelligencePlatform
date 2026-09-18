import os
import sys
import joblib

from src.utils.exception import CustomException
from src.utils.logger import logger


class ModelPersistence:

    def save_model(
        self,
        model,
        file_path: str
    ) -> str:

        try:

            logger.info(
                f"Saving model to: {file_path}"
            )

            directory = os.path.dirname(
                file_path
            )

            os.makedirs(
                directory,
                exist_ok=True
            )

            joblib.dump(
                model,
                file_path
            )

            logger.info(
                "Model saved successfully."
            )

            return file_path

        except Exception as e:

            logger.error(
                "Model saving failed."
            )

            raise CustomException(
                e,
                sys
            )

    def load_model(
        self,
        file_path: str
    ):

        try:

            logger.info(
                f"Loading model from: {file_path}"
            )

            if not os.path.exists(
                file_path
            ):

                raise FileNotFoundError(
                    f"Model file not found: "
                    f"{file_path}"
                )

            model = joblib.load(
                file_path
            )

            logger.info(
                "Model loaded successfully."
            )

            return model

        except Exception as e:

            logger.error(
                "Model loading failed."
            )

            raise CustomException(
                e,
                sys
            )