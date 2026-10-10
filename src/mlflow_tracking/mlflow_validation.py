from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient

from src.mlflow_tracking.config import (
    MLFLOW_TRACKING_URI,
    CHURN_EXPERIMENT,
    CLV_EXPERIMENT,
    RECOMMENDATION_EXPERIMENT,
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)


# ============================================================
# OUTPUT PATHS
# ============================================================

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "mlflow"
)

FINAL_REPORT_PATH = (
    OUTPUT_DIRECTORY
    / "mlflow_final_report.json"
)

RUN_SUMMARY_PATH = (
    OUTPUT_DIRECTORY
    / "mlflow_run_summary.csv"
)


# ============================================================
# EXPECTED PRODUCTION RUN NAMES
# ============================================================

EXPECTED_PRODUCTION_RUNS = {

    CHURN_EXPERIMENT:
        "Churn - Final Production Model",

    CLV_EXPERIMENT:
        "CLV - Final Production Model",

    RECOMMENDATION_EXPERIMENT:
        "Recommendation - Final Production Model",
}


# ============================================================
# EXPECTED EXPERIMENTS
# ============================================================

EXPECTED_EXPERIMENTS = [
    CHURN_EXPERIMENT,
    CLV_EXPERIMENT,
    RECOMMENDATION_EXPERIMENT,
]


# ============================================================
# MLFLOW VALIDATOR
# ============================================================

class MLflowValidation:

    def __init__(
        self,
        tracking_uri: str = MLFLOW_TRACKING_URI,
    ):

        self.tracking_uri = tracking_uri

        mlflow.set_tracking_uri(
            self.tracking_uri
        )

        self.client = MlflowClient(
            tracking_uri=
                self.tracking_uri
        )

        OUTPUT_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # GET EXPERIMENT
    # ========================================================

    def get_experiment(
        self,
        experiment_name: str,
    ):

        return mlflow.get_experiment_by_name(
            experiment_name
        )

    # ========================================================
    # GET RUNS
    # ========================================================

    def get_runs(
        self,
        experiment_id: str,
    ) -> pd.DataFrame:

        runs = mlflow.search_runs(
            experiment_ids=[
                experiment_id
            ],
            output_format="pandas",
        )

        if runs is None:

            return pd.DataFrame()

        return runs

    # ========================================================
    # RUN NAME COLUMN
    # ========================================================

    @staticmethod
    def get_run_name(
        row: pd.Series,
    ) -> str:

        run_name_columns = [
            "tags.mlflow.runName",
            "tags.mlflow.run_name",
            "run_name",
        ]

        for column in run_name_columns:

            if (
                column in row.index
                and
                pd.notna(
                    row[column]
                )
            ):

                return str(
                    row[column]
                )

        return "Unnamed Run"

    # ========================================================
    # RUN ID
    # ========================================================

    @staticmethod
    def get_run_id(
        row: pd.Series,
    ) -> str:

        candidates = [
            "run_id",
            "run_uuid",
        ]

        for column in candidates:

            if (
                column in row.index
                and
                pd.notna(
                    row[column]
                )
            ):

                return str(
                    row[column]
                )

        return ""

    # ========================================================
    # RUN STATUS
    # ========================================================

    @staticmethod
    def get_run_status(
        row: pd.Series,
    ) -> str:

        if (
            "status" in row.index
            and
            pd.notna(
                row["status"]
            )
        ):

            return str(
                row["status"]
            )

        return "UNKNOWN"

    # ========================================================
    # PRODUCTION RUN PRESENT
    # ========================================================

    @staticmethod
    def production_run_present(
        runs: pd.DataFrame,
        expected_run_name: str,
    ) -> bool:

        if runs.empty:

            return False

        possible_columns = [
            "tags.mlflow.runName",
            "tags.mlflow.run_name",
        ]

        for column in possible_columns:

            if column in runs.columns:

                run_names = (
                    runs[column]
                    .dropna()
                    .astype(str)
                    .tolist()
                )

                if expected_run_name in (
                    run_names
                ):

                    return True

        return False

    # ========================================================
    # FINISHED RUN COUNT
    # ========================================================

    @staticmethod
    def finished_run_count(
        runs: pd.DataFrame,
    ) -> int:

        if runs.empty:

            return 0

        if "status" not in runs.columns:

            return 0

        return int(
            (
                runs["status"]
                .astype(str)
                .str.upper()
                ==
                "FINISHED"
            ).sum()
        )

    # ========================================================
    # FAILED RUN COUNT
    # ========================================================

    @staticmethod
    def failed_run_count(
        runs: pd.DataFrame,
    ) -> int:

        if runs.empty:

            return 0

        if "status" not in runs.columns:

            return 0

        return int(
            (
                runs["status"]
                .astype(str)
                .str.upper()
                ==
                "FAILED"
            ).sum()
        )

    # ========================================================
    # BUILD RUN SUMMARY
    # ========================================================

    def build_run_summary(
        self,
        experiment_name: str,
        experiment_id: str,
        runs: pd.DataFrame,
    ) -> list[dict[str, Any]]:

        summary_rows = []

        if runs.empty:

            return summary_rows

        for _, row in runs.iterrows():

            summary_rows.append(
                {
                    "Experiment":
                        experiment_name,

                    "Experiment ID":
                        experiment_id,

                    "Run ID":
                        self.get_run_id(
                            row
                        ),

                    "Run Name":
                        self.get_run_name(
                            row
                        ),

                    "Status":
                        self.get_run_status(
                            row
                        ),
                }
            )

        return summary_rows

    # ========================================================
    # VALIDATE EXPERIMENT
    # ========================================================

    def validate_experiment(
        self,
        experiment_name: str,
    ) -> dict[str, Any]:

        experiment = (
            self.get_experiment(
                experiment_name
            )
        )

        if experiment is None:

            return {
                "experiment_name":
                    experiment_name,

                "exists":
                    False,

                "experiment_id":
                    None,

                "artifact_location":
                    None,

                "run_count":
                    0,

                "finished_run_count":
                    0,

                "failed_run_count":
                    0,

                "production_run_expected":
                    EXPECTED_PRODUCTION_RUNS.get(
                        experiment_name
                    ),

                "production_run_present":
                    False,

                "status":
                    "MISSING",
            }

        runs = self.get_runs(
            experiment.experiment_id
        )

        expected_production_run = (
            EXPECTED_PRODUCTION_RUNS.get(
                experiment_name
            )
        )

        production_present = False

        if expected_production_run:

            production_present = (
                self.production_run_present(
                    runs=
                        runs,

                    expected_run_name=
                        expected_production_run,
                )
            )

        run_count = len(
            runs
        )

        finished_count = (
            self.finished_run_count(
                runs
            )
        )

        failed_count = (
            self.failed_run_count(
                runs
            )
        )

        if (
            run_count > 0
            and
            production_present
            and
            failed_count == 0
        ):

            status = "PASS"

        else:

            status = "REVIEW_REQUIRED"

        return {
            "experiment_name":
                experiment_name,

            "exists":
                True,

            "experiment_id":
                experiment.experiment_id,

            "artifact_location":
                experiment.artifact_location,

            "run_count":
                run_count,

            "finished_run_count":
                finished_count,

            "failed_run_count":
                failed_count,

            "production_run_expected":
                expected_production_run,

            "production_run_present":
                production_present,

            "status":
                status,
        }

    # ========================================================
    # VALIDATE ALL
    # ========================================================

    def validate_all(
        self,
    ) -> tuple[
        dict[str, Any],
        pd.DataFrame,
    ]:

        experiment_results = {}

        run_summary_rows = []

        for experiment_name in (
            EXPECTED_EXPERIMENTS
        ):

            result = (
                self.validate_experiment(
                    experiment_name
                )
            )

            experiment_results[
                experiment_name
            ] = result

            if result[
                "exists"
            ]:

                runs = self.get_runs(
                    result[
                        "experiment_id"
                    ]
                )

                run_summary_rows.extend(
                    self.build_run_summary(
                        experiment_name=
                            experiment_name,

                        experiment_id=
                            result[
                                "experiment_id"
                            ],

                        runs=
                            runs,
                    )
                )

        summary_df = pd.DataFrame(
            run_summary_rows
        )

        all_experiments_exist = all(
            result[
                "exists"
            ]
            for result
            in experiment_results.values()
        )

        all_have_runs = all(
            result[
                "run_count"
            ] > 0
            for result
            in experiment_results.values()
        )

        all_production_runs = all(
            result[
                "production_run_present"
            ]
            for result
            in experiment_results.values()
        )

        total_failed_runs = sum(
            result[
                "failed_run_count"
            ]
            for result
            in experiment_results.values()
        )

        overall_status = (
            "PASS"
            if (
                all_experiments_exist
                and
                all_have_runs
                and
                all_production_runs
                and
                total_failed_runs == 0
            )
            else
            "REVIEW_REQUIRED"
        )

        report = {
            "phase":
                "Phase 9 — MLflow",

            "status":
                overall_status,

            "tracking_uri":
                self.tracking_uri,

            "experiments":
                experiment_results,

            "validation_summary": {
                "expected_experiment_count":
                    len(
                        EXPECTED_EXPERIMENTS
                    ),

                "all_experiments_exist":
                    all_experiments_exist,

                "all_experiments_have_runs":
                    all_have_runs,

                "all_production_runs_present":
                    all_production_runs,

                "total_failed_runs":
                    total_failed_runs,

                "overall_status":
                    overall_status,
            },

            "project_tracking_design": {
                "churn":
                    {
                        "task":
                            "binary classification",

                        "main_metrics":
                            [
                                "Precision",
                                "Recall",
                                "F1",
                                "ROC-AUC",
                            ],
                    },

                "clv":
                    {
                        "task":
                            "90-day revenue regression",

                        "main_metrics":
                            [
                                "MAE",
                                "RMSE",
                                "R2",
                            ],

                        "target_transform":
                            "log1p",

                        "prediction_scale":
                            "original revenue scale",
                    },

                "recommendation":
                    {
                        "task":
                            "top-N recommendation",

                        "main_metrics":
                            [
                                "Precision@K",
                                "Recall@K",
                                "HitRate@K",
                                "CatalogCoverage",
                            ],

                        "evaluation":
                            "temporal holdout",
                    },
            },

            "known_limitations": [
                (
                    "Recommendation temporal evaluation "
                    "was used for development and model "
                    "selection and should not be described "
                    "as an untouched final test set."
                ),
                (
                    "Recommendation scores are ranking "
                    "signals rather than calibrated "
                    "purchase probabilities."
                ),
                (
                    "CLV represents predicted future "
                    "90-day revenue rather than literal "
                    "customer lifetime value."
                ),
                (
                    "Local SHAP explanations are based "
                    "on a precomputed customer sample."
                ),
            ],
        }

        return (
            report,
            summary_df,
        )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    report: dict[str, Any],
    summary_df: pd.DataFrame,
) -> None:

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        FINAL_REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    summary_df.to_csv(
        RUN_SUMMARY_PATH,
        index=False,
    )