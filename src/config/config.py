import os

class ProjectConfig:

    PROJECT_NAME = "Customer_Intelligence_Platform"
    DATA_DIR = os.path.join("data")
    RAW_DATA_DIR = os.path.join(DATA_DIR,"raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR,"processed")
    MODEL_DIR = os.path.join("models")
    LOG_DIR = os.path.join("logs")

    ARTIFACTS_DIR = os.path.join("artifacts")
