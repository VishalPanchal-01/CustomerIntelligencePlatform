import json
from pathlib import Path

from src.mlflow_tracking.config import (
    CLV_EXPERIMENT,
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
# PRODUCTION CLV ARTIFACT PATHS
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "clv"
    / "clv_model.pkl"
)

VALUE_BANDS_PATH = (
    PROJECT_ROOT
    / "models"
    / "clv"
    / "clv_value_bands.json"
)

MODEL_REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "clv"
    / "clv_model_report.json"
)

FINAL_METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "clv"
    / "final"
    / "clv_final_model_metadata.json"
)

FEATURE_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "clv"
    / "final"
    / "clv_feature_importance.csv"
)

FINAL_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "clv"
    / "final"
    / "clv_final_test_predictions.csv"
)


# ============================================================
# HELPER
# ============================================================

def get_scalar_metadata(
    metadata: dict,
) -> dict:

    scalar_values = {}

    for key, value in (
        metadata.items()
    ):

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):

            scalar_values[
                f"metadata_{key}"
            ] = value

    return scalar_values


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "========================================"
    )

    print(
        "TRACK FINAL CLV PRODUCTION MODEL"
    )

    print(
        "========================================"
        "\n"
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Final CLV model not found: "
            f"{MODEL_PATH}"
        )

    tracker = MLflowTracker(
        CLV_EXPERIMENT
    )

    with tracker.start_run(
        run_name=
            "CLV - Final Production Model",

        tags={
            "phase":
                "clv",

            "task":
                "regression",

            "model_stage":
                "production",

            "target":
                "FutureRevenue",

            "business_target":
                "future_90_day_revenue",

            "forecast_horizon_days":
                "90",

            "prediction_scale":
                "original_revenue",

            "target_transform":
                "log1p",

            "inverse_transform":
                "expm1",

            "double_expm1":
                "false",
        },

    ) as run:

        # ====================================================
        # CORE PARAMETERS
        # ====================================================

        tracker.log_params(
            {
                "feature_count":
                    6,

                "customer_id_column":
                    "Customer ID",

                "target_column":
                    "FutureRevenue",

                "forecast_horizon_days":
                    90,

                "target_transform":
                    "log1p",

                "inverse_transform":
                    "expm1",

                "model_predict_returns":
                    "original_revenue_scale",

                "negative_prediction_handling":
                    "clip_to_zero",
            }
        )

        # ====================================================
        # PRODUCTION MODEL ARTIFACT
        # ====================================================

        tracker.log_artifact(
            str(
                MODEL_PATH
            ),
            artifact_path=
                "production_model",
        )

        print(
            "Production model logged."
        )

        # ====================================================
        # VALUE BANDS
        # ====================================================

        if VALUE_BANDS_PATH.exists():

            tracker.log_artifact(
                str(
                    VALUE_BANDS_PATH
                ),
                artifact_path=
                    "value_bands",
            )

            print(
                "CLV value bands logged."
            )

        else:

            print(
                "CLV value bands not found. "
                "Skipping."
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
                "CLV model report logged."
            )

        else:

            print(
                "CLV model report not found. "
                "Skipping."
            )

        # ====================================================
        # FINAL MODEL METADATA
        # ====================================================

        if FINAL_METADATA_PATH.exists():

            tracker.log_artifact(
                str(
                    FINAL_METADATA_PATH
                ),
                artifact_path=
                    "metadata",
            )

            try:

                with open(
                    FINAL_METADATA_PATH,
                    "r",
                    encoding="utf-8",
                ) as file:

                    metadata = json.load(
                        file
                    )

                scalar_metadata = (
                    get_scalar_metadata(
                        metadata
                    )
                )

                if scalar_metadata:

                    tracker.log_params(
                        scalar_metadata
                    )

                tracker.log_tags(
                    {
                        "final_metadata_available":
                            "true"
                    }
                )

                print(
                    "Final CLV metadata logged."
                )

            except Exception as error:

                tracker.log_tags(
                    {
                        "final_metadata_available":
                            "false"
                    }
                )

                print(
                    "Could not read final "
                    f"metadata: {error}"
                )

        else:

            tracker.log_tags(
                {
                    "final_metadata_available":
                        "false"
                }
            )

            print(
                "Final CLV metadata not found. "
                "Skipping."
            )

        # ====================================================
        # FEATURE IMPORTANCE
        # ====================================================

        if FEATURE_IMPORTANCE_PATH.exists():

            tracker.log_artifact(
                str(
                    FEATURE_IMPORTANCE_PATH
                ),
                artifact_path=
                    "feature_importance",
            )

            print(
                "Feature importance logged."
            )

        else:

            print(
                "Feature importance file "
                "not found. Skipping."
            )

        # ====================================================
        # FINAL TEST PREDICTIONS
        # ====================================================

        if FINAL_PREDICTIONS_PATH.exists():

            tracker.log_artifact(
                str(
                    FINAL_PREDICTIONS_PATH
                ),
                artifact_path=
                    "predictions",
            )

            print(
                "Final test predictions logged."
            )

        else:

            print(
                "Final test predictions "
                "not found. Skipping."
            )

        # ====================================================
        # COMPLETE
        # ====================================================

        print(
            "\nFinal CLV production "
            "tracking completed."
        )

        print(
            f"MLflow Run ID: "
            f"{run.info.run_id}"
        )


if __name__ == "__main__":

    main()