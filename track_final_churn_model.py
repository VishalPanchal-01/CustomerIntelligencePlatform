import json
from pathlib import Path

import mlflow

from src.mlflow_tracking.config import (
    CHURN_EXPERIMENT
)

from src.mlflow_tracking.tracker import (
    MLflowTracker
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn"
    / "churn_model.pkl"
)

MODEL_REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "churn"
    / "churn_model_report.json"
)


def main():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Final churn model not found: "
            f"{MODEL_PATH}"
        )

    tracker = MLflowTracker(
        CHURN_EXPERIMENT
    )

    with tracker.start_run(
        run_name=
            "Churn - Final Production Model",

        tags={
            "phase":
                "churn",

            "model_stage":
                "production",

            "target":
                "Churn",

            "positive_class":
                "1"
        }
    ) as run:

        tracker.log_artifact(
            str(
                MODEL_PATH
            ),
            artifact_path=
                "production_model"
        )

        if MODEL_REPORT_PATH.exists():

            tracker.log_artifact(
                str(
                    MODEL_REPORT_PATH
                ),
                artifact_path=
                    "production_report"
            )

            try:

                with open(
                    MODEL_REPORT_PATH,
                    "r",
                    encoding="utf-8"
                ) as file:

                    report = json.load(
                        file
                    )

                tracker.log_tags(
                    {
                        "model_report_available":
                            True
                    }
                )

            except Exception:

                pass

        tracker.log_params(
            {
                "feature_count":
                    6,

                "customer_id_column":
                    "Customer ID",

                "target_column":
                    "Churn"
            }
        )

        print(
            "Final churn production artifact "
            "logged successfully."
        )

        print(
            f"Run ID: {run.info.run_id}"
        )


if __name__ == "__main__":

    main()