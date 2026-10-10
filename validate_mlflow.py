import json
import os
from pathlib import Path

from src.mlflow_tracking.config import (
    CHURN_EXPERIMENT,
    CLV_EXPERIMENT,
    RECOMMENDATION_EXPERIMENT,
    MLFLOW_TRACKING_URI
)

from src.mlflow_tracking.tracker import (
    MLflowTracker
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

REPORT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "mlflow"
)

REPORT_PATH = (
    REPORT_DIR
    / "mlflow_foundation_report.json"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "========================================"
    )

    print(
        "MLFLOW FOUNDATION VALIDATION"
    )

    print(
        "========================================"
        "\n"
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    experiments = [
        CHURN_EXPERIMENT,
        CLV_EXPERIMENT,
        RECOMMENDATION_EXPERIMENT
    ]

    experiment_status = {}

    for experiment_name in experiments:

        tracker = MLflowTracker(
            experiment_name
        )

        info = (
            tracker.tracking_info()
        )

        experiment_status[
            experiment_name
        ] = info

        print(
            f"Experiment: {experiment_name}"
        )

        print(
            f"Experiment ID: "
            f"{info['experiment_id']}"
        )

        print(
            f"Tracking URI: "
            f"{info['tracking_uri']}"
        )

        print()

    all_created = all(
        result[
            "experiment_id"
        ]
        is not None
        for result
        in experiment_status.values()
    )

    status = (
        "PASS"
        if all_created
        else
        "REVIEW_REQUIRED"
    )

    report = {

        "phase":
            "Phase 9 — MLflow",

        "step":
            "MLflow Foundation",

        "status":
            status,

        "tracking_uri":
            MLFLOW_TRACKING_URI,

        "experiments":
            experiment_status,

        "validation": {

            "churn_experiment_created":
                (
                    CHURN_EXPERIMENT
                    in experiment_status
                    and
                    experiment_status[
                        CHURN_EXPERIMENT
                    ][
                        "experiment_id"
                    ]
                    is not None
                ),

            "clv_experiment_created":
                (
                    CLV_EXPERIMENT
                    in experiment_status
                    and
                    experiment_status[
                        CLV_EXPERIMENT
                    ][
                        "experiment_id"
                    ]
                    is not None
                ),

            "recommendation_experiment_created":
                (
                    RECOMMENDATION_EXPERIMENT
                    in experiment_status
                    and
                    experiment_status[
                        RECOMMENDATION_EXPERIMENT
                    ][
                        "experiment_id"
                    ]
                    is not None
                )
        }
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(
        f"Overall Status: {status}"
    )

    print(
        "\nReport saved:"
    )

    print(
        REPORT_PATH
    )


if __name__ == "__main__":

    main()