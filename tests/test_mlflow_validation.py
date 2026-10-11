from pathlib import Path

import mlflow

from src.mlflow_tracking.config import (
    CHURN_EXPERIMENT,
    CLV_EXPERIMENT,
    RECOMMENDATION_EXPERIMENT,
)

from src.mlflow_tracking.mlflow_validation import (
    EXPECTED_EXPERIMENTS,
    EXPECTED_PRODUCTION_RUNS,
    MLflowValidation,
)


# ============================================================
# EXPECTED EXPERIMENT CONFIGURATION
# ============================================================

def test_expected_experiments():

    assert (
        CHURN_EXPERIMENT
        in EXPECTED_EXPERIMENTS
    )

    assert (
        CLV_EXPERIMENT
        in EXPECTED_EXPERIMENTS
    )

    assert (
        RECOMMENDATION_EXPERIMENT
        in EXPECTED_EXPERIMENTS
    )


# ============================================================
# EXPECTED PRODUCTION RUN NAMES
# ============================================================

def test_expected_production_runs():

    assert (
        EXPECTED_PRODUCTION_RUNS[
            CHURN_EXPERIMENT
        ]
        ==
        "Churn - Final Production Model"
    )

    assert (
        EXPECTED_PRODUCTION_RUNS[
            CLV_EXPERIMENT
        ]
        ==
        "CLV - Final Production Model"
    )

    assert (
        EXPECTED_PRODUCTION_RUNS[
            RECOMMENDATION_EXPERIMENT
        ]
        ==
        "Recommendation - Final Production Model"
    )


# ============================================================
# VALIDATOR INITIALIZATION
# ============================================================

def test_validator_initialization(
    tmp_path: Path,
):

    tracking_directory = (
        tmp_path
        / "mlruns"
    )

    tracking_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracking_uri = (
        tracking_directory
        .resolve()
        .as_uri()
    )

    validator = MLflowValidation(
        tracking_uri=
            tracking_uri
    )

    assert (
        validator.tracking_uri
        ==
        tracking_uri
    )

    assert (
        validator.client
        is not None
    )


# ============================================================
# MISSING EXPERIMENT HANDLING
# ============================================================

def test_missing_experiment(
    tmp_path: Path,
):

    tracking_directory = (
        tmp_path
        / "mlruns"
    )

    tracking_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracking_uri = (
        tracking_directory
        .resolve()
        .as_uri()
    )

    validator = MLflowValidation(
        tracking_uri=
            tracking_uri
    )

    result = (
        validator.validate_experiment(
            "Experiment That Does Not Exist"
        )
    )

    assert (
        result[
            "exists"
        ]
        is False
    )

    assert (
        result[
            "run_count"
        ]
        ==
        0
    )

    assert (
        result[
            "status"
        ]
        ==
        "MISSING"
    )


# ============================================================
# EXPERIMENT WITH RUN
# ============================================================

def test_experiment_with_finished_run(
    tmp_path: Path,
):

    tracking_directory = (
        tmp_path
        / "mlruns"
    )

    tracking_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracking_uri = (
        tracking_directory
        .resolve()
        .as_uri()
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    experiment_name = (
        "CI Test Experiment"
    )

    experiment_id = (
        mlflow.create_experiment(
            experiment_name
        )
    )

    with mlflow.start_run(
        experiment_id=
            experiment_id,
        run_name=
            "CI Test Run",
    ):

        mlflow.log_param(
            "model",
            "test-model",
        )

        mlflow.log_metric(
            "score",
            0.95,
        )

    validator = MLflowValidation(
        tracking_uri=
            tracking_uri
    )

    experiment = (
        validator.get_experiment(
            experiment_name
        )
    )

    assert (
        experiment
        is not None
    )

    runs = validator.get_runs(
        experiment.experiment_id
    )

    assert len(
        runs
    ) == 1

    assert (
        validator.finished_run_count(
            runs
        )
        ==
        1
    )

    assert (
        validator.failed_run_count(
            runs
        )
        ==
        0
    )