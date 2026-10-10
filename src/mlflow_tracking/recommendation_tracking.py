import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

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
    .parents[2]
)


# ============================================================
# DEFAULT ARTIFACT PATHS
# ============================================================

DEFAULT_COMPARISON_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "recommendation"
    / "final"
    / "recommendation_final_comparison.csv"
)

DEFAULT_SELECTION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "recommendation"
    / "final"
    / "recommendation_final_selection.json"
)

DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "recommendation"
    / "recommendation_model_report.json"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "mlflow"
    / "recommendation"
)


# ============================================================
# RECOMMENDATION MLFLOW TRACKING
# ============================================================

class RecommendationMLflowTracking:

    # ========================================================
    # POSSIBLE COLUMN NAMES
    # ========================================================

    MODEL_COLUMN_CANDIDATES = [
        "Model",
        "model",
        "Model Name",
        "model_name",
        "Recommender",
        "Algorithm",
    ]

    METRIC_COLUMN_CANDIDATES = {

        "precision_at_k": [
            "Precision@K",
            "precision@k",
            "Precision",
            "precision",
            "precision_at_k",
            "PrecisionAtK",
        ],

        "recall_at_k": [
            "Recall@K",
            "recall@k",
            "Recall",
            "recall",
            "recall_at_k",
            "RecallAtK",
        ],

        "hit_rate_at_k": [
            "HitRate@K",
            "Hit Rate@K",
            "HitRate",
            "Hit Rate",
            "hit_rate",
            "hit_rate_at_k",
            "hitrate_at_k",
        ],

        "catalog_coverage": [
            "CatalogCoverage",
            "Catalog Coverage",
            "Coverage",
            "coverage",
            "catalog_coverage",
        ],
    }

    TOP_K_COLUMN_CANDIDATES = [
        "K",
        "TopK",
        "Top K",
        "top_k",
        "k",
    ]

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        comparison_path: str | Path = DEFAULT_COMPARISON_PATH,
        selection_path: str | Path = DEFAULT_SELECTION_PATH,
        report_path: str | Path = DEFAULT_REPORT_PATH,
        output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
        default_top_k: int = 10,
    ):

        self.comparison_path = Path(
            comparison_path
        )

        self.selection_path = Path(
            selection_path
        )

        self.report_path = Path(
            report_path
        )

        self.output_directory = Path(
            output_directory
        )

        self.default_top_k = (
            default_top_k
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.tracker = MLflowTracker(
            RECOMMENDATION_EXPERIMENT
        )

    # ========================================================
    # FIND COLUMN
    # ========================================================

    @staticmethod
    def find_column(
        dataframe: pd.DataFrame,
        candidates: list[str],
    ) -> str | None:

        for column in candidates:

            if column in dataframe.columns:

                return column

        lowered_columns = {
            str(column).lower():
                column
            for column
            in dataframe.columns
        }

        for candidate in candidates:

            lowered_candidate = (
                candidate.lower()
            )

            if lowered_candidate in (
                lowered_columns
            ):

                return lowered_columns[
                    lowered_candidate
                ]

        return None

    # ========================================================
    # LOAD COMPARISON
    # ========================================================

    def load_comparison(
        self,
    ) -> pd.DataFrame:

        if not self.comparison_path.exists():

            raise FileNotFoundError(
                "Recommendation comparison file "
                "not found: "
                f"{self.comparison_path}"
            )

        df = pd.read_csv(
            self.comparison_path
        )

        if df.empty:

            raise ValueError(
                "Recommendation comparison "
                "file is empty."
            )

        return df

    # ========================================================
    # LOAD FINAL SELECTION
    # ========================================================

    def load_selection(
        self,
    ) -> dict[str, Any]:

        if not self.selection_path.exists():

            return {}

        try:

            with open(
                self.selection_path,
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
                "Could not read recommendation "
                f"selection JSON: {error}"
            )

        return {}

    # ========================================================
    # MODEL COLUMN
    # ========================================================

    def get_model_column(
        self,
        df: pd.DataFrame,
    ) -> str:

        model_column = (
            self.find_column(
                dataframe=df,
                candidates=
                    self.MODEL_COLUMN_CANDIDATES,
            )
        )

        if model_column is None:

            raise ValueError(
                "Could not identify the model "
                "name column in recommendation "
                "comparison CSV.\n"
                f"Available columns: "
                f"{list(df.columns)}"
            )

        return model_column

    # ========================================================
    # METRIC COLUMN MAP
    # ========================================================

    def get_metric_columns(
        self,
        df: pd.DataFrame,
    ) -> dict[str, str]:

        metric_columns = {}

        for metric_name, candidates in (
            self.METRIC_COLUMN_CANDIDATES.items()
        ):

            column = self.find_column(
                dataframe=df,
                candidates=candidates,
            )

            if column is not None:

                metric_columns[
                    metric_name
                ] = column

        if not metric_columns:

            raise ValueError(
                "No supported recommendation "
                "metric columns were found.\n"
                f"Available columns: "
                f"{list(df.columns)}"
            )

        return metric_columns

    # ========================================================
    # TOP K
    # ========================================================

    def get_top_k(
        self,
        row: pd.Series,
        df: pd.DataFrame,
    ) -> int:

        top_k_column = (
            self.find_column(
                dataframe=df,
                candidates=
                    self.TOP_K_COLUMN_CANDIDATES,
            )
        )

        if top_k_column is None:

            return self.default_top_k

        try:

            value = int(
                row[
                    top_k_column
                ]
            )

            if value > 0:

                return value

        except (
            TypeError,
            ValueError,
        ):

            pass

        return self.default_top_k

    # ========================================================
    # CLEAN MODEL NAME
    # ========================================================

    @staticmethod
    def normalize_model_name(
        value: Any,
    ) -> str:

        if value is None:

            return "Unknown Recommender"

        name = str(
            value
        ).strip()

        if not name:

            return "Unknown Recommender"

        return name

    # ========================================================
    # GET FINAL SELECTED MODEL NAME
    # ========================================================

    @staticmethod
    def selected_model_name(
        selection: dict[str, Any],
    ) -> str | None:

        candidate_keys = [
            "selected_model",
            "final_model",
            "best_model",
            "model",
            "model_name",
            "selected_recommender",
            "winner",
        ]

        for key in candidate_keys:

            value = selection.get(
                key
            )

            if isinstance(
                value,
                str,
            ) and value.strip():

                return value.strip()

        nested_candidates = [
            "selection",
            "final_selection",
            "best",
        ]

        for nested_key in (
            nested_candidates
        ):

            nested_value = selection.get(
                nested_key
            )

            if isinstance(
                nested_value,
                dict,
            ):

                nested_name = (
                    RecommendationMLflowTracking
                    .selected_model_name(
                        nested_value
                    )
                )

                if nested_name:

                    return nested_name

        return None

    # ========================================================
    # ROW METRICS
    # ========================================================

    @staticmethod
    def extract_metrics(
        row: pd.Series,
        metric_columns: dict[str, str],
    ) -> dict[str, float]:

        metrics = {}

        for metric_name, column in (
            metric_columns.items()
        ):

            value = row[
                column
            ]

            try:

                numeric_value = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                continue

            if pd.isna(
                numeric_value
            ):

                continue

            metrics[
                metric_name
            ] = numeric_value

        return metrics

    # ========================================================
    # SAVE NORMALIZED COMPARISON
    # ========================================================

    def save_normalized_comparison(
        self,
        df: pd.DataFrame,
        model_column: str,
        metric_columns: dict[str, str],
    ) -> Path:

        rows = []

        top_k_column = (
            self.find_column(
                dataframe=df,
                candidates=
                    self.TOP_K_COLUMN_CANDIDATES,
            )
        )

        for _, row in df.iterrows():

            output_row = {
                "Model":
                    self.normalize_model_name(
                        row[
                            model_column
                        ]
                    )
            }

            if top_k_column:

                output_row[
                    "K"
                ] = row[
                    top_k_column
                ]

            else:

                output_row[
                    "K"
                ] = self.default_top_k

            for metric_name, column in (
                metric_columns.items()
            ):

                output_row[
                    metric_name
                ] = row[
                    column
                ]

            rows.append(
                output_row
            )

        normalized_df = (
            pd.DataFrame(
                rows
            )
        )

        output_path = (
            self.output_directory
            /
            "recommendation_mlflow_comparison.csv"
        )

        normalized_df.to_csv(
            output_path,
            index=False,
        )

        return output_path

    # ========================================================
    # SAVE EVALUATION METADATA
    # ========================================================

    def save_evaluation_metadata(
        self,
        model_count: int,
        selected_model: str | None,
        metric_names: list[str],
    ) -> Path:

        metadata = {
            "evaluation_strategy":
                (
                    "Temporal holdout based on "
                    "latest full invoice."
                ),

            "benchmark_role":
                (
                    "Development and model-selection "
                    "benchmark."
                ),

            "untouched_final_test_set":
                False,

            "important_limitation":
                (
                    "Recommendation evaluation uses "
                    "the temporal benchmark for model "
                    "comparison and selection. It should "
                    "not be described as an untouched "
                    "final recommendation test set."
                ),

            "default_top_k":
                self.default_top_k,

            "model_count":
                model_count,

            "selected_model":
                selected_model,

            "metrics":
                metric_names,

            "selection_priority":
                [
                    "Recall@K",
                    "HitRate@K",
                    "Precision@K",
                    "CatalogCoverage",
                ],

            "recommendation_score_meaning":
                (
                    "Ranking signal, not a calibrated "
                    "probability."
                ),

            "cold_start_limitation":
                (
                    "Cold-start support depends on "
                    "the persisted recommendation "
                    "implementation and is not assumed "
                    "to be universal."
                ),
        }

        output_path = (
            self.output_directory
            /
            "recommendation_evaluation_metadata.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
            )

        return output_path

    # ========================================================
    # TRACK ONE EXISTING MODEL RESULT
    # ========================================================

    def track_result(
        self,
        model_name: str,
        metrics: dict[str, float],
        top_k: int,
        selected_model: str | None,
        metadata_path: Path,
    ) -> dict[str, Any]:

        is_selected = False

        if selected_model is not None:

            is_selected = (
                model_name.strip().lower()
                ==
                selected_model.strip().lower()
            )

        with self.tracker.start_run(
            run_name=
                f"Recommendation - {model_name}",

            tags={
                "phase":
                    "recommendation",

                "task":
                    "top_n_recommendation",

                "model_name":
                    model_name,

                "evaluation":
                    "temporal_holdout",

                "model_selected":
                    str(
                        is_selected
                    ).lower(),

                "recommendation_score":
                    "ranking_signal",
            },

        ) as run:

            self.tracker.log_params(
                {
                    "top_k":
                        top_k,

                    "evaluation_strategy":
                        "temporal_holdout",

                    "selection_primary_metric":
                        "recall_at_k",

                    "selection_secondary_metric":
                        "hit_rate_at_k",

                    "selection_tertiary_metric":
                        "precision_at_k",

                    "selection_fourth_metric":
                        "catalog_coverage",

                    "untouched_final_test":
                        False,
                }
            )

            self.tracker.log_metrics(
                metrics
            )

            self.tracker.log_artifact(
                str(
                    metadata_path
                ),
                artifact_path=
                    "evaluation_metadata",
            )

            return {
                "run_id":
                    run.info.run_id,

                "model":
                    model_name,

                "selected":
                    is_selected,

                "top_k":
                    top_k,

                **metrics,
            }

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
    ) -> pd.DataFrame:

        print(
            "Loading recommendation "
            "comparison results..."
        )

        comparison_df = (
            self.load_comparison()
        )

        print(
            "Comparison file:"
        )

        print(
            self.comparison_path
        )

        print(
            "\nAvailable columns:"
        )

        print(
            list(
                comparison_df.columns
            )
        )

        # ----------------------------------------------------
        # Identify schema
        # ----------------------------------------------------

        model_column = (
            self.get_model_column(
                comparison_df
            )
        )

        metric_columns = (
            self.get_metric_columns(
                comparison_df
            )
        )

        # ----------------------------------------------------
        # Load selected model
        # ----------------------------------------------------

        selection = (
            self.load_selection()
        )

        selected_model = (
            self.selected_model_name(
                selection
            )
        )

        print(
            "\nSelected production "
            "candidate from metadata:"
        )

        print(
            selected_model
            if selected_model
            else
            "Not detected"
        )

        # ----------------------------------------------------
        # Save normalized comparison
        # ----------------------------------------------------

        normalized_path = (
            self.save_normalized_comparison(
                df=comparison_df,
                model_column=
                    model_column,
                metric_columns=
                    metric_columns,
            )
        )

        # ----------------------------------------------------
        # Save evaluation metadata
        # ----------------------------------------------------

        metadata_path = (
            self.save_evaluation_metadata(
                model_count=
                    len(
                        comparison_df
                    ),

                selected_model=
                    selected_model,

                metric_names=
                    list(
                        metric_columns.keys()
                    ),
            )
        )

        results = []

        # ----------------------------------------------------
        # One MLflow run per evaluated recommender
        # ----------------------------------------------------

        for _, row in (
            comparison_df.iterrows()
        ):

            model_name = (
                self.normalize_model_name(
                    row[
                        model_column
                    ]
                )
            )

            metrics = (
                self.extract_metrics(
                    row=row,
                    metric_columns=
                        metric_columns,
                )
            )

            top_k = (
                self.get_top_k(
                    row=row,
                    df=comparison_df,
                )
            )

            print(
                "\n"
                "========================================"
            )

            print(
                "Tracking recommender: "
                f"{model_name}"
            )

            print(
                "========================================"
            )

            if not metrics:

                print(
                    "No numeric evaluation "
                    "metrics found. Skipping."
                )

                continue

            result = (
                self.track_result(
                    model_name=
                        model_name,

                    metrics=
                        metrics,

                    top_k=
                        top_k,

                    selected_model=
                        selected_model,

                    metadata_path=
                        metadata_path,
                )
            )

            results.append(
                result
            )

        results_df = pd.DataFrame(
            results
        )

        if not results_df.empty:

            sort_columns = []

            ascending = []

            if (
                "recall_at_k"
                in results_df.columns
            ):

                sort_columns.append(
                    "recall_at_k"
                )

                ascending.append(
                    False
                )

            if (
                "hit_rate_at_k"
                in results_df.columns
            ):

                sort_columns.append(
                    "hit_rate_at_k"
                )

                ascending.append(
                    False
                )

            if (
                "precision_at_k"
                in results_df.columns
            ):

                sort_columns.append(
                    "precision_at_k"
                )

                ascending.append(
                    False
                )

            if (
                "catalog_coverage"
                in results_df.columns
            ):

                sort_columns.append(
                    "catalog_coverage"
                )

                ascending.append(
                    False
                )

            if sort_columns:

                results_df = (
                    results_df
                    .sort_values(
                        by=sort_columns,
                        ascending=ascending,
                    )
                    .reset_index(
                        drop=True
                    )
                )

        results_path = (
            self.output_directory
            /
            "recommendation_mlflow_runs.csv"
        )

        results_df.to_csv(
            results_path,
            index=False,
        )

        print(
            "\nNormalized comparison:"
        )

        print(
            normalized_path
        )

        print(
            "\nMLflow run summary:"
        )

        print(
            results_path
        )

        return results_df