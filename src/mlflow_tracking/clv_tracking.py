import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.base import clone

from sklearn.ensemble import (
    RandomForestRegressor
)

from sklearn.linear_model import (
    LinearRegression
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import (
    KFold,
    cross_validate,
    train_test_split
)

from sklearn.compose import (
    TransformedTargetRegressor
)

from src.mlflow_tracking.config import (
    CLV_EXPERIMENT
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
# PATHS
# ============================================================

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "clv"
    / "customer_clv_dataset.csv"
)

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "mlflow"
    / "clv"
)


# ============================================================
# CLV TRACKING
# ============================================================

class CLVMLflowTracking:

    FEATURE_COLUMNS = [
        "Recency",
        "Frequency",
        "Monetary",
        "TotalItems",
        "AverageOrderValue",
        "Tenure"
    ]

    TARGET_COLUMN = (
        "FutureRevenue"
    )

    CUSTOMER_ID = (
        "Customer ID"
    )


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
            CLV_EXPERIMENT
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
                "CLV dataset not found: "
                f"{self.dataset_path}"
            )

        df = pd.read_csv(
            self.dataset_path
        )

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

        required_columns = (
            self.FEATURE_COLUMNS
            +
            [
                self.TARGET_COLUMN
            ]
        )

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:

            raise ValueError(
                "Required CLV columns are missing: "
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
            pd.to_numeric(
                df[
                    self.TARGET_COLUMN
                ],
                errors="coerce"
            )
        )

        for feature in self.FEATURE_COLUMNS:

            X[
                feature
            ] = pd.to_numeric(
                X[
                    feature
                ],
                errors="coerce"
            )

        invalid_feature_rows = (
            X
            .isnull()
            .any(
                axis=1
            )
        )

        invalid_target_rows = (
            y.isnull()
        )

        if (
            invalid_feature_rows.any()
            or
            invalid_target_rows.any()
        ):

            raise ValueError(
                "CLV dataset contains missing "
                "or non-numeric values."
            )

        X_values = X.to_numpy(
            dtype=float
        )

        y_values = y.to_numpy(
            dtype=float
        )

        if not np.isfinite(
            X_values
        ).all():

            raise ValueError(
                "CLV features contain "
                "infinite values."
            )

        if not np.isfinite(
            y_values
        ).all():

            raise ValueError(
                "CLV target contains "
                "infinite values."
            )

        if (
            y < 0
        ).any():

            raise ValueError(
                "FutureRevenue cannot be negative."
            )

        return X, y


    # ========================================================
    # MODELS
    # ========================================================

    def get_models(
        self
    ) -> dict:

        linear_regression = (
            TransformedTargetRegressor(
                regressor=
                    LinearRegression(),

                func=
                    np.log1p,

                inverse_func=
                    np.expm1
            )
        )

        random_forest = (
            TransformedTargetRegressor(
                regressor=
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=
                            self.random_state,
                        n_jobs=-1
                    ),

                func=
                    np.log1p,

                inverse_func=
                    np.expm1
            )
        )

        return {

            "Linear Regression":
                linear_regression,

            "Random Forest":
                random_forest
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

        cv = KFold(
            n_splits=
                self.n_splits,

            shuffle=True,

            random_state=
                self.random_state
        )

        scoring = {

            "mae":
                "neg_mean_absolute_error",

            "mse":
                "neg_mean_squared_error",

            "r2":
                "r2"
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

        mae_values = (
            -scores[
                "test_mae"
            ]
        )

        mse_values = (
            -scores[
                "test_mse"
            ]
        )

        rmse_values = (
            np.sqrt(
                mse_values
            )
        )

        r2_values = scores[
            "test_r2"
        ]

        return {

            "cv_mae_mean":
                float(
                    np.mean(
                        mae_values
                    )
                ),

            "cv_mae_std":
                float(
                    np.std(
                        mae_values
                    )
                ),

            "cv_rmse_mean":
                float(
                    np.mean(
                        rmse_values
                    )
                ),

            "cv_rmse_std":
                float(
                    np.std(
                        rmse_values
                    )
                ),

            "cv_r2_mean":
                float(
                    np.mean(
                        r2_values
                    )
                ),

            "cv_r2_std":
                float(
                    np.std(
                        r2_values
                    )
                )
        }


    # ========================================================
    # HOLDOUT EVALUATION
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

        predictions = np.asarray(
            predictions,
            dtype=float
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # TransformedTargetRegressor.predict()
        # already performs inverse_func=np.expm1.
        #
        # DO NOT apply np.expm1() here.
        # ----------------------------------------------------

        predictions = np.clip(
            predictions,
            0.0,
            None
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mse
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        return {

            "test_mae":
                float(
                    mae
                ),

            "test_rmse":
                float(
                    rmse
                ),

            "test_r2":
                float(
                    r2
                )
        }


    # ========================================================
    # PARAMETERS
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
            / "clv_feature_metadata.json"
        )

        metadata = {

            "customer_id_column":
                self.CUSTOMER_ID,

            "target":
                self.TARGET_COLUMN,

            "target_meaning":
                (
                    "Future gross positive purchase "
                    "revenue over the next 90 days."
                ),

            "business_label":
                "Predicted 90-Day Revenue",

            "features":
                self.FEATURE_COLUMNS,

            "feature_count":
                len(
                    self.FEATURE_COLUMNS
                ),

            "target_transformation":
                "log1p",

            "inverse_transformation":
                "expm1",

            "prediction_scale":
                (
                    "Original revenue scale returned "
                    "directly by model.predict()."
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
    # SAVE PREDICTIONS
    # ========================================================

    def save_predictions(
        self,
        model_name: str,
        model,
        X_test,
        y_test
    ) -> Path:

        predictions = np.asarray(
            model.predict(
                X_test
            ),
            dtype=float
        )

        # Do not apply expm1 again.
        predictions = np.clip(
            predictions,
            0.0,
            None
        )

        output = X_test.copy()

        output[
            "Actual FutureRevenue"
        ] = (
            y_test.to_numpy()
        )

        output[
            "Predicted FutureRevenue"
        ] = predictions

        output[
            "Absolute Error"
        ] = np.abs(
            output[
                "Actual FutureRevenue"
            ]
            -
            output[
                "Predicted FutureRevenue"
            ]
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
    # TRACK MODEL
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

        with self.tracker.start_run(

            run_name=
                f"CLV - {model_name}",

            tags={
                "phase":
                    "clv",

                "task":
                    "regression",

                "target":
                    self.TARGET_COLUMN,

                "business_target":
                    "future_90_day_revenue",

                "model_name":
                    model_name,

                "target_transform":
                    "log1p"
            }

        ) as run:

            self.tracker.log_params(
                {
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
                        ),

                    "target_transform":
                        "log1p",

                    "inverse_transform":
                        "expm1",

                    "prediction_output_scale":
                        "original_revenue"
                }
            )

            self.tracker.log_params(
                self.model_parameters(
                    model
                )
            )

            # ------------------------------------------------
            # CV only on training portion
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
            # Fit
            # ------------------------------------------------

            fitted_model = clone(
                model
            )

            fitted_model.fit(
                X_train,
                y_train
            )

            # ------------------------------------------------
            # Holdout
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
            # Log model
            # ------------------------------------------------

            mlflow.sklearn.log_model(
                sk_model=
                    fitted_model,

                name=
                    "model"
            )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            self.tracker.log_artifact(
                str(
                    feature_metadata_path
                ),
                artifact_path=
                    "metadata"
            )

            # ------------------------------------------------
            # Predictions
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
    # RUN ALL
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

                random_state=
                    self.random_state
            )
        )

        metadata_path = (
            self.save_feature_metadata()
        )

        results = []

        for model_name, model in (
            self.get_models().items()
        ):

            print(
                f"\nTracking: {model_name}"
            )

            result = (
                self.track_model(
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
                        metadata_path
                )
            )

            results.append(
                result
            )

        comparison = pd.DataFrame(
            results
        )

        comparison_path = (
            self.output_directory
            / "clv_mlflow_comparison.csv"
        )

        comparison.to_csv(
            comparison_path,
            index=False
        )

        return comparison