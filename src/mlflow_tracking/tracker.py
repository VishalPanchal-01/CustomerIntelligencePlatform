from contextlib import contextmanager
from typing import Any

import mlflow
import numpy as np

from src.mlflow_tracking.config import (
    MLFLOW_TRACKING_URI
)


class MLflowTracker:

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        experiment_name: str
    ):

        self.experiment_name = (
            experiment_name
        )

        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        mlflow.set_experiment(
            experiment_name
        )


    # =========================================================
    # CLEAN VALUE
    # =========================================================

    @staticmethod
    def clean_value(
        value: Any
    ):

        if isinstance(
            value,
            (
                np.integer,
                np.floating
            )
        ):

            return value.item()

        return value


    # =========================================================
    # LOG PARAMETERS
    # =========================================================

    def log_params(
        self,
        params: dict
    ) -> None:

        cleaned = {}

        for key, value in (
            params.items()
        ):

            if value is None:

                continue

            cleaned[
                str(key)
            ] = self.clean_value(
                value
            )

        if cleaned:

            mlflow.log_params(
                cleaned
            )


    # =========================================================
    # LOG METRICS
    # =========================================================

    def log_metrics(
        self,
        metrics: dict,
        step: int | None = None
    ) -> None:

        cleaned = {}

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
                ValueError
            ):

                continue

            if not np.isfinite(
                numeric_value
            ):

                continue

            cleaned[
                str(key)
            ] = numeric_value

        if cleaned:

            mlflow.log_metrics(
                cleaned,
                step=step
            )


    # =========================================================
    # LOG TAGS
    # =========================================================

    def log_tags(
        self,
        tags: dict
    ) -> None:

        cleaned = {}

        for key, value in (
            tags.items()
        ):

            if value is None:

                continue

            cleaned[
                str(key)
            ] = str(
                value
            )

        if cleaned:

            mlflow.set_tags(
                cleaned
            )


    # =========================================================
    # LOG ARTIFACT
    # =========================================================

    @staticmethod
    def log_artifact(
        path: str,
        artifact_path: str | None = None
    ) -> None:

        mlflow.log_artifact(
            path,
            artifact_path=
                artifact_path
        )


    # =========================================================
    # LOG ARTIFACTS DIRECTORY
    # =========================================================

    @staticmethod
    def log_artifacts(
        directory: str,
        artifact_path: str | None = None
    ) -> None:

        mlflow.log_artifacts(
            directory,
            artifact_path=
                artifact_path
        )


    # =========================================================
    # ACTIVE RUN
    # =========================================================

    @contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict | None = None
    ):

        with mlflow.start_run(
            run_name=
                run_name
        ) as run:

            if tags:

                self.log_tags(
                    tags
                )

            yield run


    # =========================================================
    # TRACKING INFORMATION
    # =========================================================

    def tracking_info(
        self
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
                )
        }