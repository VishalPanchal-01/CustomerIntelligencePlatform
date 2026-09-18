import os
import sys
import joblib

from src.utils.exception import CustomException
from src.utils.logger import logger


class SegmentationPersistence:

    def save_artifact(
        self,
        artifact,
        file_path: str
    ) -> str:

        try:

            logger.info(
                f"Saving segmentation artifact to: "
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
                artifact,
                file_path
            )

            logger.info(
                "Segmentation artifact saved successfully."
            )

            return file_path

        except Exception as e:

            logger.error(
                "Segmentation artifact saving failed."
            )

            raise CustomException(
                e,
                sys
            )

    def load_artifact(
        self,
        file_path: str
    ):

        try:

            logger.info(
                f"Loading segmentation artifact from: "
                f"{file_path}"
            )

            if not os.path.exists(
                file_path
            ):

                raise FileNotFoundError(
                    f"Artifact not found: "
                    f"{file_path}"
                )

            artifact = joblib.load(
                file_path
            )

            logger.info(
                "Segmentation artifact loaded successfully."
            )

            return artifact

        except Exception as e:

            logger.error(
                "Segmentation artifact loading failed."
            )

            raise CustomException(
                e,
                sys
            )