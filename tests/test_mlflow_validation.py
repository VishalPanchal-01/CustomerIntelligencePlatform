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


def test_mlflow_validator_initialization():

    validator = (
        MLflowValidation()
    )

    assert (
        validator.tracking_uri
        is not None
    )

    assert (
        validator.client
        is not None
    )


def test_validate_all_structure():

    validator = (
        MLflowValidation()
    )

    report, summary_df = (
        validator.validate_all()
    )

    assert (
        "status"
        in report
    )

    assert (
        "experiments"
        in report
    )

    assert (
        "validation_summary"
        in report
    )

    assert (
        CHURN_EXPERIMENT
        in report[
            "experiments"
        ]
    )

    assert (
        CLV_EXPERIMENT
        in report[
            "experiments"
        ]
    )

    assert (
        RECOMMENDATION_EXPERIMENT
        in report[
            "experiments"
        ]
    )

    assert (
        summary_df
        is not None
    )