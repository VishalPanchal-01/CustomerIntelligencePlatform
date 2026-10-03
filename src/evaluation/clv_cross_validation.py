import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.training.clv_linear_model import (
    CLVLinearRegressionModel
)

from src.training.clv_random_forest_model import (
    CLVRandomForestModel
)

from src.training.clv_gradient_boosting_model import (
    CLVGradientBoostingModel
)

from src.utils.exception import CustomException
from src.utils.logger import logger


class CLVCrossValidator:

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42
    ):

        self.n_splits = n_splits
        self.random_state = random_state

    def evaluate_models(
        self,
        X: pd.DataFrame,
        y_raw: pd.Series,
        y_log: pd.Series
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Starting CLV cross-validation."
            )

            # ---------------------------------
            # Validate input
            # ---------------------------------

            if X.empty:

                raise ValueError(
                    "CLV feature dataset is empty."
                )

            if len(X) != len(y_raw):

                raise ValueError(
                    "X and raw target lengths do not match."
                )

            if len(X) != len(y_log):

                raise ValueError(
                    "X and log target lengths do not match."
                )

            if self.n_splits < 2:

                raise ValueError(
                    "Cross-validation requires at least 2 folds."
                )

            if self.n_splits > len(X):

                raise ValueError(
                    "Number of folds cannot exceed number of samples."
                )

            # ---------------------------------
            # K-Fold strategy
            # ---------------------------------

            cv = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state
            )

            # ---------------------------------
            # Model configurations
            # ---------------------------------

            configurations = [
                {
                    "model_name":
                        "Linear Regression",

                    "target_type":
                        "Raw"
                },

                {
                    "model_name":
                        "Linear Regression",

                    "target_type":
                        "Log"
                },

                {
                    "model_name":
                        "Random Forest",

                    "target_type":
                        "Raw"
                },

                {
                    "model_name":
                        "Random Forest",

                    "target_type":
                        "Log"
                },

                {
                    "model_name":
                        "Gradient Boosting",

                    "target_type":
                        "Raw"
                },

                {
                    "model_name":
                        "Gradient Boosting",

                    "target_type":
                        "Log"
                }
            ]

            results = []

            # =================================
            # MODEL LOOP
            # =================================

            for configuration in configurations:

                model_name = (
                    configuration[
                        "model_name"
                    ]
                )

                target_type = (
                    configuration[
                        "target_type"
                    ]
                )

                logger.info(
                    f"Cross-validating "
                    f"{model_name} - {target_type}"
                )

                fold_mae = []
                fold_rmse = []
                fold_r2 = []

                fold_negative_predictions = []

                # =================================
                # FOLD LOOP
                # =================================

                for fold_number, (
                    train_index,
                    validation_index
                ) in enumerate(
                    cv.split(X),
                    start=1
                ):

                    logger.info(
                        f"{model_name} - "
                        f"{target_type} - "
                        f"Fold {fold_number}"
                    )

                    # ---------------------------------
                    # Feature split
                    # ---------------------------------

                    X_train_fold = (
                        X.iloc[
                            train_index
                        ]
                        .copy()
                    )

                    X_validation_fold = (
                        X.iloc[
                            validation_index
                        ]
                        .copy()
                    )

                    # ---------------------------------
                    # Raw targets
                    # ---------------------------------

                    y_raw_train_fold = (
                        y_raw.iloc[
                            train_index
                        ]
                        .copy()
                    )

                    y_raw_validation_fold = (
                        y_raw.iloc[
                            validation_index
                        ]
                        .copy()
                    )

                    # ---------------------------------
                    # Log target
                    # ---------------------------------

                    y_log_train_fold = (
                        y_log.iloc[
                            train_index
                        ]
                        .copy()
                    )

                    # ---------------------------------
                    # Build fresh model
                    # ---------------------------------

                    model = (
                        self._build_model(
                            model_name
                        )
                    )

                    # ---------------------------------
                    # Select training target
                    # ---------------------------------

                    if target_type == "Raw":

                        training_target = (
                            y_raw_train_fold
                        )

                    else:

                        training_target = (
                            y_log_train_fold
                        )

                    # ---------------------------------
                    # Train
                    # ---------------------------------

                    model.fit(
                        X_train_fold,
                        training_target
                    )

                    # ---------------------------------
                    # Predict validation fold
                    # ---------------------------------

                    predictions = (
                        model.predict(
                            X_validation_fold
                        )
                    )

                    predictions = np.asarray(
                        predictions,
                        dtype=float
                    )

                    # ---------------------------------
                    # Convert log prediction
                    # back to revenue
                    # ---------------------------------

                    if target_type == "Log":

                        predictions = (
                            np.expm1(
                                predictions
                            )
                        )

                    # ---------------------------------
                    # Count negative predictions
                    # ---------------------------------

                    negative_count = int(
                        (
                            predictions < 0
                        )
                        .sum()
                    )

                    fold_negative_predictions.append(
                        negative_count
                    )

                    # ---------------------------------
                    # Business-safe predictions
                    # ---------------------------------

                    predictions = np.maximum(
                        predictions,
                        0
                    )

                    actual = np.asarray(
                        y_raw_validation_fold,
                        dtype=float
                    )

                    # ---------------------------------
                    # Fold MAE
                    # ---------------------------------

                    mae = float(
                        mean_absolute_error(
                            actual,
                            predictions
                        )
                    )

                    # ---------------------------------
                    # Fold RMSE
                    # ---------------------------------

                    mse = float(
                        mean_squared_error(
                            actual,
                            predictions
                        )
                    )

                    rmse = float(
                        np.sqrt(
                            mse
                        )
                    )

                    # ---------------------------------
                    # Fold R²
                    # ---------------------------------

                    r2 = float(
                        r2_score(
                            actual,
                            predictions
                        )
                    )

                    fold_mae.append(
                        mae
                    )

                    fold_rmse.append(
                        rmse
                    )

                    fold_r2.append(
                        r2
                    )

                # =================================
                # AGGREGATE RESULTS
                # =================================

                result = {

                    "Model":
                        model_name,

                    "Target":
                        target_type,

                    "MAEMean":
                        float(
                            np.mean(
                                fold_mae
                            )
                        ),

                    "MAEStd":
                        float(
                            np.std(
                                fold_mae
                            )
                        ),

                    "RMSEMean":
                        float(
                            np.mean(
                                fold_rmse
                            )
                        ),

                    "RMSEStd":
                        float(
                            np.std(
                                fold_rmse
                            )
                        ),

                    "R2Mean":
                        float(
                            np.mean(
                                fold_r2
                            )
                        ),

                    "R2Std":
                        float(
                            np.std(
                                fold_r2
                            )
                        ),

                    "TotalNegativePredictions":
                        int(
                            np.sum(
                                fold_negative_predictions
                            )
                        )
                }

                results.append(
                    result
                )

            # =================================
            # FINAL REPORT
            # =================================

            report = pd.DataFrame(
                results
            )

            report = (
                report
                .sort_values(
                    by=[
                        "MAEMean",
                        "RMSEMean"
                    ],
                    ascending=[
                        True,
                        True
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            logger.info(
                f"CLV cross-validation completed:\n"
                f"{report}"
            )

            return report

        except Exception as e:

            logger.error(
                "CLV cross-validation failed."
            )

            raise CustomException(
                e,
                sys
            )

    def _build_model(
        self,
        model_name: str
    ):

        if model_name == "Linear Regression":

            return (
                CLVLinearRegressionModel()
                .build_model()
            )

        if model_name == "Random Forest":

            return (
                CLVRandomForestModel(
                    n_estimators=200,
                    random_state=42
                )
                .build_model()
            )

        if model_name == "Gradient Boosting":

            return (
                CLVGradientBoostingModel(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                )
                .build_model()
            )

        raise ValueError(
            f"Unsupported CLV model: "
            f"{model_name}"
        )