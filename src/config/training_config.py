import os
from datetime import datetime


class TrainingPipelineConfig:

    PIPELINE_NAME = "customer_intelligence_training"

    ARTIFACT_DIR = "artifacts"

    CURRENT_TIME_STAMP = datetime.now().strftime(
        "%m_%d_%Y_%H_%M_%S"
    )

    PIPELINE_ARTIFACT_DIR = os.path.join(
        ARTIFACT_DIR,
        CURRENT_TIME_STAMP
    )

    MODEL_DIR = os.path.join(
        PIPELINE_ARTIFACT_DIR,
        "models"
    )

    METRICS_DIR = os.path.join(
        PIPELINE_ARTIFACT_DIR,
        "metrics"
    )

    REPORT_DIR = os.path.join(
        PIPELINE_ARTIFACT_DIR,
        "reports"
    )