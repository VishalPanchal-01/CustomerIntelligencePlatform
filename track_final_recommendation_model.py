import json
from pathlib import Path

from src.mlflow_tracking.config import (
    RECOMMENDATION_EXPERIMENT,
)

from src.mlflow_tracking.tracker import (
    MLflowTracker,
)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)


# ============================================================
# PRODUCTION ARTIFACT PATHS
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
    / "recommender.pkl"
)

MODEL_METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "recommendation"
    / "recommender_metadata.json"
)

FINAL_COMPARISON_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "recommendation"
    / "final"
    / "recommendation_final_comparison.csv"
)

FINAL_SELECTION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "recommendation"
    / "final"
    / "recommendation_final_selection.json"
)

BATCH_RECOMMENDATIONS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "recommendation"
    / "predictions"
    / "batch_recommendations.csv"
)

MODEL_REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "recommendation"
    / "recommendation_model_report.json"
)


# ============================================================
# EXTRACT SCALAR PARAMETERS
# ============================================================

def extract_scalar_values(
    data: dict,
    prefix: str = "",
) -> dict:

    result = {}

    for key, value in (
        data.items()
    ):

        parameter_name = (
            f"{prefix}{key}"
        )

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):

            result[
                parameter_name
            ] = value

    return result


# ============================================================
# LOAD JSON
# ============================================================

def load_json(
    path: Path,
) -> dict:

    if not path.exists():

        return {}

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        if isinstance(
            data,
            dict,
        ):

            return data

    except Exception as error:

        print(
            f"Could not read {path}: "
            f"{error}"
        )

    return {}


# ============================================================
# FIND SELECTED MODEL
# ============================================================

def find_selected_model(
    data: dict,
) -> str | None:

    keys = [
        "selected_model",
        "final_model",
        "best_model",
        "model",
        "model_name",
        "selected_recommender",
        "winner",
    ]

    for key in keys:

        value = data.get(
            key
        )

        if isinstance(
            value,
            str,
        ) and value.strip():

            return value.strip()

    for value in data.values():

        if isinstance(
            value,
            dict,
        ):

            result = (
                find_selected_model(
                    value
                )
            )

            if result:

                return result

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "========================================"
    )

    print(
        "TRACK FINAL RECOMMENDATION MODEL"
    )

    print(
        "========================================"
        "\n"
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Final recommendation model "
            "not found: "
            f"{MODEL_PATH}"
        )

    selection_metadata = (
        load_json(
            FINAL_SELECTION_PATH
        )
    )

    recommender_metadata = (
        load_json(
            MODEL_METADATA_PATH
        )
    )

    selected_model = (
        find_selected_model(
            selection_metadata
        )
    )

    tracker = MLflowTracker(
        RECOMMENDATION_EXPERIMENT
    )

    with tracker.start_run(
        run_name=
            "Recommendation - Final Production Model",

        tags={
            "phase":
                "recommendation",

            "task":
                "top_n_recommendation",

            "model_stage":
                "production",

            "selected_model":
                (
                    selected_model
                    if selected_model
                    else "unknown"
                ),

            "evaluation_strategy":
                "temporal_holdout",

            "recommendation_score":
                "ranking_signal",

            "cold_start":
                (
                    "implementation_dependent"
                ),

            "untouched_final_test":
                "false",
        },

    ) as run:

        # ====================================================
        # CORE PRODUCTION PARAMETERS
        # ====================================================

        tracker.log_params(
            {
                "top_k":
                    10,

                "evaluation_strategy":
                    "temporal_holdout",

                "selection_primary_metric":
                    "Recall@K",

                "selection_secondary_metric":
                    "HitRate@K",

                "selection_tertiary_metric":
                    "Precision@K",

                "selection_fourth_metric":
                    "CatalogCoverage",

                "recommendation_score_type":
                    "ranking_signal",

                "model_retrained_on":
                    "full_history",

                "untouched_final_test":
                    False,
            }
        )

        # ====================================================
        # MODEL
        # ====================================================

        tracker.log_artifact(
            str(
                MODEL_PATH
            ),
            artifact_path=
                "production_model",
        )

        print(
            "Production recommender logged."
        )

        # ====================================================
        # MODEL METADATA
        # ====================================================

        if MODEL_METADATA_PATH.exists():

            tracker.log_artifact(
                str(
                    MODEL_METADATA_PATH
                ),
                artifact_path=
                    "model_metadata",
            )

            scalar_metadata = (
                extract_scalar_values(
                    recommender_metadata,
                    prefix="metadata_",
                )
            )

            if scalar_metadata:

                tracker.log_params(
                    scalar_metadata
                )

            print(
                "Recommender metadata logged."
            )

        else:

            print(
                "Recommender metadata not "
                "found. Skipping."
            )

        # ====================================================
        # FINAL MODEL COMPARISON
        # ====================================================

        if FINAL_COMPARISON_PATH.exists():

            tracker.log_artifact(
                str(
                    FINAL_COMPARISON_PATH
                ),
                artifact_path=
                    "evaluation",
            )

            print(
                "Final comparison logged."
            )

        else:

            print(
                "Final comparison file not "
                "found. Skipping."
            )

        # ====================================================
        # FINAL SELECTION
        # ====================================================

        if FINAL_SELECTION_PATH.exists():

            tracker.log_artifact(
                str(
                    FINAL_SELECTION_PATH
                ),
                artifact_path=
                    "selection",
            )

            selection_scalars = (
                extract_scalar_values(
                    selection_metadata,
                    prefix="selection_",
                )
            )

            if selection_scalars:

                tracker.log_params(
                    selection_scalars
                )

            print(
                "Final selection metadata "
                "logged."
            )

        else:

            print(
                "Final selection metadata "
                "not found. Skipping."
            )

        # ====================================================
        # BATCH RECOMMENDATIONS
        # ====================================================

        if BATCH_RECOMMENDATIONS_PATH.exists():

            tracker.log_artifact(
                str(
                    BATCH_RECOMMENDATIONS_PATH
                ),
                artifact_path=
                    "predictions",
            )

            print(
                "Batch recommendations logged."
            )

        else:

            print(
                "Batch recommendation artifact "
                "not found. Skipping."
            )

        # ====================================================
        # MODEL REPORT
        # ====================================================

        if MODEL_REPORT_PATH.exists():

            tracker.log_artifact(
                str(
                    MODEL_REPORT_PATH
                ),
                artifact_path=
                    "production_report",
            )

            print(
                "Recommendation model report "
                "logged."
            )

        else:

            print(
                "Recommendation model report "
                "not found. Skipping."
            )

        # ====================================================
        # COMPLETION
        # ====================================================

        print(
            "\nFinal recommendation "
            "production tracking completed."
        )

        print(
            f"Selected model: "
            f"{selected_model}"
        )

        print(
            f"MLflow Run ID: "
            f"{run.info.run_id}"
        )


if __name__ == "__main__":

    main()