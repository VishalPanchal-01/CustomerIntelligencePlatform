from contextlib import contextmanager
from typing import Any

import mlflow
import numpy as np

from src.mlflow_tracking.config import (
    MLFLOW_TRACKING_URI,
)


# ============================================================
# MLFLOW TRACKER
# ============================================================

class MLflowTracker:

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        experiment_name: str,
    ):

        self.experiment_name = (
            experiment_name
        )

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        mlflow.set_experiment(
            self.experiment_name
        )

    # ========================================================
    # CLEAN VALUE
    # ========================================================

    @staticmethod
    def clean_value(
        value: Any,
    ) -> Any:

        if isinstance(
            value,
            (
                np.integer,
                np.floating,
            ),
        ):

            return value.item()

        return value

    # ========================================================
    # LOG PARAMETERS
    # ========================================================

    def log_params(
        self,
        params: dict,
    ) -> None:

        cleaned_params = {}

        for key, value in (
            params.items()
        ):

            if value is None:

                continue

            cleaned_params[
                str(key)
            ] = self.clean_value(
                value
            )

        if cleaned_params:

            mlflow.log_params(
                cleaned_params
            )

    # ========================================================
    # LOG METRICS
    # ========================================================

    def log_metrics(
        self,
        metrics: dict,
        step: int | None = None,
    ) -> None:

        cleaned_metrics = {}

        for key, value in (
            metrics.items()
        ):

            if value is None:

                continue

            try:

                numeric_value = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            if not np.isfinite(
                numeric_value
            ):

                continue

            cleaned_metrics[
                str(key)
            ] = numeric_value

        if cleaned_metrics:

            mlflow.log_metrics(
                cleaned_metrics,
                step=step,
            )

    # ========================================================
    # LOG TAGS
    # ========================================================

    def log_tags(
        self,
        tags: dict,
    ) -> None:

        cleaned_tags = {}

        for key, value in (
            tags.items()
        ):

            if value is None:

                continue

            cleaned_tags[
                str(key)
            ] = str(
                value
            )

        if cleaned_tags:

            mlflow.set_tags(
                cleaned_tags
            )

    # ========================================================
    # LOG SINGLE ARTIFACT
    # ========================================================

    @staticmethod
    def log_artifact(
        path: str,
        artifact_path: str | None = None,
    ) -> None:

        mlflow.log_artifact(
            path,
            artifact_path=
                artifact_path,
        )

    # ========================================================
    # LOG ARTIFACT DIRECTORY
    # ========================================================

    @staticmethod
    def log_artifacts(
        directory: str,
        artifact_path: str | None = None,
    ) -> None:

        mlflow.log_artifacts(
            directory,
            artifact_path=
                artifact_path,
        )

    # ========================================================
    # START RUN
    # ========================================================

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict | None = None,
    ):

        with mlflow.start_run(
            run_name=
                run_name,
        ) as run:

            if tags:

                self.log_tags(
                    tags
                )

            yield run

    # ========================================================
    # TRACKING INFO
    # ========================================================

    def tracking_info(
        self,
    ) -> dict:

        experiment = (
            mlflow.get_experiment_by_name(
                self.experiment_name
            )
        )

        return {
            "tracking_uri":
                mlflow.get_tracking_uri(),

            "experiment_name":
                self.experiment_name,

            "experiment_id":
                (
                    experiment.experiment_id
                    if experiment
                    else None
                ),

            "artifact_location":
                (
                    experiment.artifact_location
                    if experiment
                    else None
                ),
        }