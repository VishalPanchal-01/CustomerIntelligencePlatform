import os
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# MLFLOW DATABASE
# ============================================================

MLFLOW_DB_PATH = (
    PROJECT_ROOT
    / "mlflow.db"
)


# ============================================================
# TRACKING URI
# ============================================================

DEFAULT_TRACKING_URI = (
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)

MLFLOW_TRACKING_URI = (
    os.getenv(
        "MLFLOW_TRACKING_URI",
        DEFAULT_TRACKING_URI
    )
)


# ============================================================
# ARTIFACT DIRECTORY
# ============================================================

MLFLOW_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "mlartifacts"
)


# ============================================================
# EXPERIMENT NAMES
# ============================================================

CHURN_EXPERIMENT = (
    "Customer Churn Prediction"
)

CLV_EXPERIMENT = (
    "90-Day Revenue Prediction"
)

RECOMMENDATION_EXPERIMENT = (
    "Next Product Recommendation"
)


# ============================================================
# MODEL NAMES
# ============================================================

CHURN_MODEL_NAME = (
    "customer-churn-model"
)

CLV_MODEL_NAME = (
    "customer-clv-model"
)

RECOMMENDATION_MODEL_NAME = (
    "next-product-recommender"
)