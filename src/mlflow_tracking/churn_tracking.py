import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.base import clone

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split
)

from sklearn.pipeline import (
    Pipeline
)

from sklearn.preprocessing import (
    StandardScaler
)

from src.mlflow_tracking.config import (
    CHURN_EXPERIMENT
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
    .parents[2]
)


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "churn"
    / "customer_churn_dataset.csv"
)

DEFAULT_FINAL_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn"
    / "churn_model.pkl"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "mlflow"
    / "churn"
)


# ============================================================
# CHURN MLFLOW TRACKER
# ============================================================

class ChurnMLflowTracking:

    FEATURE_COLUMNS = [
        "Recency",
        "Frequency",
        "Monetary",
        "TotalItems",
        "AverageOrderValue",
        "Tenure"
    ]

    TARGET_COLUMN = "Churn"

    CUSTOMER_ID = "Customer ID"


    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        output_directory: str | Path = DEFAULT_OUTPUT_DIRECTORY,
        random_state: int = 42,
        test_size: float = 0.20,
        n_splits: int = 5
    ):

        self.dataset_path = Path(
            dataset_path
        )

        self.output_directory = Path(
            output_directory
        )

        self.random_state = (
            random_state
        )

        self.test_size = (
            test_size
        )

        self.n_splits = (
            n_splits
        )

        self.tracker = MLflowTracker(
            CHURN_EXPERIMENT
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True
        )


    # ========================================================
    # LOAD DATASET
    # ========================================================

    def load_dataset(
        self
    ) -> pd.DataFrame:

        if not self.dataset_path.exists():

            raise FileNotFoundError(
                "Churn dataset not found: "
                f"{self.dataset_path}"
            )

        df = pd.read_csv(
            self.dataset_path
        )

        # ----------------------------------------------------
        # Backward compatibility only
        # ----------------------------------------------------

        if (
            "CustomerID" in df.columns
            and
            self.CUSTOMER_ID not in df.columns
        ):

            df = df.rename(
                columns={
                    "CustomerID":
                        self.CUSTOMER_ID
                }
            )

        missing = [
            column
            for column in (
                self.FEATURE_COLUMNS
                +
                [
                    self.TARGET_COLUMN
                ]
            )
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                "Required churn columns are missing: "
                f"{missing}"
            )

        return df


    # ========================================================
    # PREPARE DATA
    # ========================================================

    def prepare_data(
        self,
        df: pd.DataFrame
    ):

        X = (
            df[
                self.FEATURE_COLUMNS
            ]
            .copy()
        )

        y = (
            df[
                self.TARGET_COLUMN
            ]
            .astype(int)
            .copy()
        )

        # ----------------------------------------------------
        # Numeric validation
        # ----------------------------------------------------

        for feature in self.FEATURE_COLUMNS:

            X[
                feature
            ] = pd.to_numeric(
                X[
                    feature
                ],
                errors="coerce"
            )

        invalid_rows = (
            X
            .isnull()
            .any(
                axis=1
            )
        )

        if invalid_rows.any():

            raise ValueError(
                "Churn feature dataset contains "
                "missing or non-numeric feature values."
            )

        values = X.to_numpy(
            dtype=float
        )

        if not np.isfinite(
            values
        ).all():

            raise ValueError(
                "Churn dataset contains "
                "infinite feature values."
            )

        unique_target = set(
            y.unique()
        )

        if not unique_target.issubset(
            {
                0,
                1
            }
        ):

            raise ValueError(
                "Churn target must contain "
                "binary values 0 and 1."
            )

        return X, y


    # ========================================================
    # MODEL DEFINITIONS
    # ========================================================

    def get_models(
        self
    ) -> dict:

        logistic_model = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=
                            self.random_state
                    )
                )
            ]
        )

        random_forest_model = (
            RandomForestClassifier(
                n_estimators=300,
                random_state=
                    self.random_state,
                n_jobs=-1,
                class_weight="balanced"
            )
        )

        return {
            "Logistic Regression":
                logistic_model,

            "Random Forest":
                random_forest_model
        }


    # ========================================================
    # CROSS VALIDATION
    # ========================================================

    def cross_validate_model(
        self,
        model,
        X_train,
        y_train
    ) -> dict:

        cv = StratifiedKFold(
            n_splits=
                self.n_splits,

            shuffle=True,

            random_state=
                self.random_state
        )

        scoring = {
            "accuracy":
                "accuracy",

            "precision":
                "precision",

            "recall":
                "recall",

            "f1":
                "f1",

            "roc_auc":
                "roc_auc"
        }

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        metrics = {}

        for metric_name in scoring:

            values = scores[
                f"test_{metric_name}"
            ]

            metrics[
                f"cv_{metric_name}_mean"
            ] = float(
                np.mean(
                    values
                )
            )

            metrics[
                f"cv_{metric_name}_std"
            ] = float(
                np.std(
                    values
                )
            )

        return metrics


    # ========================================================
    # HOLDOUT METRICS
    # ========================================================

    @staticmethod
    def evaluate_holdout(
        model,
        X_test,
        y_test
    ) -> dict:

        predictions = model.predict(
            X_test
        )

        metrics = {
            "test_accuracy":
                accuracy_score(
                    y_test,
                    predictions
                ),

            "test_precision":
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

            "test_recall":
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0
                ),

            "test_f1":
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0
                )
        }

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model.predict_proba(
                    X_test
                )
            )[:, 1]

            metrics[
                "test_roc_auc"
            ] = roc_auc_score(
                y_test,
                probabilities
            )

        return {
            key:
                float(
                    value
                )
            for key, value
            in metrics.items()
        }


    # ========================================================
    # MODEL PARAMS
    # ========================================================

    @staticmethod
    def model_parameters(
        model
    ) -> dict:

        parameters = (
            model.get_params(
                deep=True
            )
        )

        cleaned = {}

        for key, value in (
            parameters.items()
        ):

            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool
                )
            ):

                cleaned[
                    key
                ] = value

            elif value is None:

                cleaned[
                    key
                ] = "None"

        return cleaned


    # ========================================================
    # FEATURE METADATA
    # ========================================================

    def save_feature_metadata(
        self
    ) -> Path:

        path = (
            self.output_directory
            / "churn_feature_metadata.json"
        )

        metadata = {
            "customer_id_column":
                self.CUSTOMER_ID,

            "target":
                self.TARGET_COLUMN,

            "target_meaning": {
                "0":
                    "Customer purchased in future window.",

                "1":
                    (
                        "Customer did not purchase "
                        "in future window."
                    )
            },

            "features":
                self.FEATURE_COLUMNS,

            "feature_count":
                len(
                    self.FEATURE_COLUMNS
                )
        }

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4
            )

        return path


    # ========================================================
    # PREDICTIONS
    # ========================================================

    def save_predictions(
        self,
        model_name: str,
        model,
        X_test,
        y_test
    ) -> Path:

        prediction = model.predict(
            X_test
        )

        output = X_test.copy()

        output[
            "Actual Churn"
        ] = (
            y_test
            .to_numpy()
        )

        output[
            "Predicted Churn"
        ] = prediction

        if hasattr(
            model,
            "predict_proba"
        ):

            output[
                "Churn Probability"
            ] = (
                model.predict_proba(
                    X_test
                )[:, 1]
            )

        safe_name = (
            model_name
            .lower()
            .replace(
                " ",
                "_"
            )
        )

        path = (
            self.output_directory
            /
            f"{safe_name}_test_predictions.csv"
        )

        output.to_csv(
            path,
            index=False
        )

        return path


    # ========================================================
    # TRACK SINGLE MODEL
    # ========================================================

    def track_model(
        self,
        model_name: str,
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        feature_metadata_path: Path
    ) -> dict:

        run_name = (
            f"Churn - {model_name}"
        )

        with self.tracker.start_run(
            run_name=
                run_name,

            tags={
                "phase":
                    "churn",

                "task":
                    "binary_classification",

                "target":
                    self.TARGET_COLUMN,

                "positive_class":
                    "1",

                "model_name":
                    model_name
            }
        ) as run:

            # ------------------------------------------------
            # Parameters
            # ------------------------------------------------

            base_params = {
                "random_state":
                    self.random_state,

                "test_size":
                    self.test_size,

                "cv_splits":
                    self.n_splits,

                "feature_count":
                    len(
                        self.FEATURE_COLUMNS
                    ),

                "train_rows":
                    len(
                        X_train
                    ),

                "test_rows":
                    len(
                        X_test
                    )
            }

            self.tracker.log_params(
                base_params
            )

            self.tracker.log_params(
                self.model_parameters(
                    model
                )
            )

            # ------------------------------------------------
            # Cross validation
            # ------------------------------------------------

            cv_metrics = (
                self.cross_validate_model(
                    model=
                        clone(
                            model
                        ),

                    X_train=
                        X_train,

                    y_train=
                        y_train
                )
            )

            self.tracker.log_metrics(
                cv_metrics
            )

            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            fitted_model = clone(
                model
            )

            fitted_model.fit(
                X_train,
                y_train
            )

            # ------------------------------------------------
            # Final holdout evaluation
            # ------------------------------------------------

            test_metrics = (
                self.evaluate_holdout(
                    fitted_model,
                    X_test,
                    y_test
                )
            )

            self.tracker.log_metrics(
                test_metrics
            )

            # ------------------------------------------------
            # Model
            # ------------------------------------------------

            mlflow.sklearn.log_model(
                sk_model=
                    fitted_model,

                name=
                    "model"
            )

            # ------------------------------------------------
            # Feature metadata
            # ------------------------------------------------

            self.tracker.log_artifact(
                str(
                    feature_metadata_path
                ),
                artifact_path=
                    "metadata"
            )

            # ------------------------------------------------
            # Test predictions
            # ------------------------------------------------

            predictions_path = (
                self.save_predictions(
                    model_name=
                        model_name,

                    model=
                        fitted_model,

                    X_test=
                        X_test,

                    y_test=
                        y_test
                )
            )

            self.tracker.log_artifact(
                str(
                    predictions_path
                ),
                artifact_path=
                    "predictions"
            )

            return {
                "run_id":
                    run.info.run_id,

                "model":
                    model_name,

                **cv_metrics,

                **test_metrics
            }


    # ========================================================
    # TRACK ALL MODELS
    # ========================================================

    def run(
        self
    ) -> pd.DataFrame:

        df = self.load_dataset()

        X, y = self.prepare_data(
            df
        )

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=
                    self.test_size,

                stratify=
                    y,

                random_state=
                    self.random_state
            )
        )

        feature_metadata_path = (
            self.save_feature_metadata()
        )

        models = self.get_models()

        results = []

        for model_name, model in (
            models.items()
        ):

            print(
                "\nTracking:"
                f" {model_name}"
            )

            result = self.track_model(
                model_name=
                    model_name,

                model=
                    model,

                X_train=
                    X_train,

                X_test=
                    X_test,

                y_train=
                    y_train,

                y_test=
                    y_test,

                feature_metadata_path=
                    feature_metadata_path
            )

            results.append(
                result
            )

        comparison = pd.DataFrame(
            results
        )

        comparison_path = (
            self.output_directory
            /
            "churn_mlflow_comparison.csv"
        )

        comparison.to_csv(
            comparison_path,
            index=False
        )

        return comparison