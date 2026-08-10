import sys

from src.config.config import ProjectConfig
from src.config.environment import PROJECT_ENV
from src.config.training_config import TrainingPipelineConfig
from src.utils.exception import CustomException
from src.utils.logger import logger


def test_foundation():

    logger.info("Starting foundation test")

    try:

        # Configuration
        assert ProjectConfig.PROJECT_NAME == "Customer_Intelligence_Platform"

        # Environment
        assert PROJECT_ENV == "development"

        # Training configuration
        assert (
            TrainingPipelineConfig.PIPELINE_NAME
            == "customer_intelligence_training"
        )

        # Exception system
        try:
            raise ValueError("Foundation test error")

        except ValueError as error:
            custom_error = CustomException(error, sys)

            assert "Foundation test error" in str(custom_error)

        logger.info("Foundation test completed successfully")

    except Exception as error:

        logger.error("Foundation test failed")

        raise CustomException(error, sys)