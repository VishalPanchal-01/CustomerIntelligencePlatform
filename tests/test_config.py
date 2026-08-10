from src.config.config import ProjectConfig
from src.config.environment import PROJECT_ENV


def test_project_config():

    assert ProjectConfig.PROJECT_NAME == "Customer_Intelligence_Platform"
    assert ProjectConfig.DATA_DIR == "data"
    assert ProjectConfig.MODEL_DIR == "models"


def test_environment():

    assert PROJECT_ENV == "development"