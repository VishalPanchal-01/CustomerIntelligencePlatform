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
# MLFLOW STORAGE
# ============================================================

MLRUNS_DIR = (
    PROJECT_ROOT
    / "mlruns"
)

MLFLOW_DATABASE_PATH = (
    PROJECT_ROOT
    / "mlflow.db"
)


# ============================================================
# DEFAULT SQLITE TRACKING URI
# ============================================================

DEFAULT_MLFLOW_TRACKING_URI = (
    "sqlite:///"
    + MLFLOW_DATABASE_PATH
    .resolve()
    .as_posix()
)


# ============================================================
# TRACKING URI
# ============================================================

MLFLOW_TRACKING_URI = (
    os.getenv(
        "MLFLOW_TRACKING_URI",
        DEFAULT_MLFLOW_TRACKING_URI,
    )
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