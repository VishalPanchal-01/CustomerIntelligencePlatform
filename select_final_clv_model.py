import json
import os

import pandas as pd

from src.preprocessing.clv_preprocessing import (
    CLVPreprocessor
)

from src.training.clv_data_split import (
    CLVDataSplitter
)

from src.training.clv_hyperparameter_tuning import (
    CLVHyperparameterTuner
)

from src.evaluation.final_clv_model_selection import (
    FinalCLVModelSelector
)

from src.analysis.clv_model_analysis import (
    CLVModelAnalysis
)

from src.utils.clv_model_persistence import (
    CLVModelPersistence
)


def make_json_serializable(
    value
):

    if isinstance(
        value,
        dict
    ):

        return {
            str(key):
                make_json_serializable(
                    item
                )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple)
    ):

        return [
            make_json_serializable(
                item
            )
            for item in value
        ]

    if hasattr(
        value,
        "item"
    ):

        return value.item()

    return value


def main():

    # =================================
    # PATHS
    # =================================

    input_path = (
        "artifacts/clv/"
        "customer_clv_dataset.csv"
    )

    output_directory = (
        "artifacts/clv/final"
    )

    model_directory = (
        "models/clv"
    )

    model_path = os.path.join(
        model_directory,
        "clv_model.pkl"
    )

    feature_importance_path = os.path.join(
        output_directory,
        "clv_feature_importance.csv"
    )

    metadata_path = os.path.join(
        output_directory,
        "clv_final_model_metadata.json"
    )

    prediction_path = os.path.join(
        output_directory,
        "clv_final_test_predictions.csv"
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    os.makedirs(
        model_directory,
        exist_ok=True
    )

    # =================================
    # LOAD DATA
    # =================================

    df = pd.read_csv(
        input_path
    )

    # =================================
    # PREPARE FEATURES
    # =================================

    preprocessor = (
        CLVPreprocessor()
    )

    (
        X,
        y_raw,
        y_log
    ) = preprocessor.prepare_features(
        df
    )

    feature_names = (
        X.columns.tolist()
    )

    # =================================
    # FINAL TRAIN / TEST SPLIT
    # =================================

    splitter = (
        CLVDataSplitter()
    )

    (
        X_train,
        X_test,
        y_raw_train,
        y_raw_test,
        y_log_train,
        y_log_test
    ) = splitter.split_data(
        X,
        y_raw,
        y_log,
        test_size=0.20,
        random_state=42
    )

    # =================================
    # TUNING
    # ONLY TRAIN DATA USED
    # =================================

    tuner = (
        CLVHyperparameterTuner(
            n_splits=5,
            random_state=42
        )
    )

    rf_raw_search = (
        tuner.tune_random_forest_raw(
            X_train=
                X_train,

            y_train=
                y_raw_train,

            n_iter=15
        )
    )

    rf_log_search = (
        tuner.tune_random_forest_log(
            X_train=
                X_train,

            y_train_raw=
                y_raw_train,

            n_iter=15
        )
    )

    gb_raw_search = (
        tuner.tune_gradient_boosting_raw(
            X_train=
                X_train,

            y_train=
                y_raw_train,

            n_iter=15
        )
    )

    gb_log_search = (
        tuner.tune_gradient_boosting_log(
            X_train=
                X_train,

            y_train_raw=
                y_raw_train,

            n_iter=15
        )
    )

    # =================================
    # CREATE CANDIDATES
    # =================================

    candidates = {

        "random_forest_raw": {

            "model_name":
                "Random Forest",

            "target_type":
                "Raw",

            "search":
                rf_raw_search
        },

        "random_forest_log": {

            "model_name":
                "Random Forest",

            "target_type":
                "Log",

            "search":
                rf_log_search
        },

        "gradient_boosting_raw": {

            "model_name":
                "Gradient Boosting",

            "target_type":
                "Raw",

            "search":
                gb_raw_search
        },

        "gradient_boosting_log": {

            "model_name":
                "Gradient Boosting",

            "target_type":
                "Log",

            "search":
                gb_log_search
        }
    }

    # =================================
    # SELECT USING CV ONLY
    # =================================

    selector = (
        FinalCLVModelSelector()
    )

    selected_candidate = (
        selector.select_best_candidate(
            candidates
        )
    )

    final_model = (
        selected_candidate[
            "best_estimator"
        ]
    )

    # =================================
    # ONLY NOW USE FINAL TEST SET
    # =================================

    final_metrics = (
        selector.evaluate_final_model(
            model=
                final_model,

            X_test=
                X_test,

            y_test=
                y_raw_test
        )
    )

    # =================================
    # FEATURE IMPORTANCE
    # =================================

    analyzer = (
        CLVModelAnalysis()
    )

    feature_importance = (
        analyzer
        .analyze_tree_importance(
            final_model,
            feature_names
        )
    )

    feature_importance.to_csv(
        feature_importance_path,
        index=False
    )

    # =================================
    # SAVE FINAL MODEL
    # =================================

    persistence = (
        CLVModelPersistence()
    )

    persistence.save_model(
        final_model,
        model_path
    )

    # =================================
    # SAVE TEST PREDICTIONS
    # =================================

    prediction_df = pd.DataFrame(
        {
            "Customer ID":
                df.loc[
                    X_test.index,
                    "Customer ID"
                ]
                .values,

            "ActualFutureRevenue":
                y_raw_test.values,

            "PredictedFutureRevenue":
                final_metrics[
                    "predictions"
                ]
        }
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    # =================================
    # FINAL METADATA
    # =================================

    metadata = {

        "model": {

            "name":
                selected_candidate[
                    "model_name"
                ],

            "target_strategy":
                selected_candidate[
                    "target_type"
                ],

            "artifact_path":
                model_path,

            "best_parameters":
                make_json_serializable(
                    selected_candidate[
                        "best_params"
                    ]
                )
        },

        "selection": {

            "criterion":
                "Lowest cross-validated MAE",

            "best_cv_mae":
                float(
                    selected_candidate[
                        "cv_mae"
                    ]
                ),

            "final_test_used_for_selection":
                False
        },

        "features":
            feature_names,

        "target": {

            "name":
                "FutureRevenue",

            "prediction_horizon_days":
                90
        },

        "final_test": {

            "customer_count":
                int(
                    len(
                        X_test
                    )
                ),

            "mae":
                float(
                    final_metrics[
                        "mae"
                    ]
                ),

            "rmse":
                float(
                    final_metrics[
                        "rmse"
                    ]
                ),

            "r2":
                float(
                    final_metrics[
                        "r2"
                    ]
                ),

            "negative_prediction_count":
                int(
                    final_metrics[
                        "negative_prediction_count"
                    ]
                ),

            "negative_prediction_percentage":
                float(
                    final_metrics[
                        "negative_prediction_percentage"
                    ]
                ),

            "actual_mean":
                float(
                    final_metrics[
                        "actual_mean"
                    ]
                ),

            "predicted_mean":
                float(
                    final_metrics[
                        "predicted_mean"
                    ]
                ),

            "actual_median":
                float(
                    final_metrics[
                        "actual_median"
                    ]
                ),

            "predicted_median":
                float(
                    final_metrics[
                        "predicted_median"
                    ]
                )
        },

        "artifacts": {

            "feature_importance":
                feature_importance_path,

            "test_predictions":
                prediction_path
        }
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # =================================
    # CONSOLE OUTPUT
    # =================================

    print(
        "\n================================"
    )

    print(
        "FINAL CLV MODEL"
    )

    print(
        "================================"
    )

    print(
        f"\nSelected Model: "
        f"{selected_candidate['model_name']}"
    )

    print(
        f"Target Strategy: "
        f"{selected_candidate['target_type']}"
    )

    print(
        f"Best CV MAE: "
        f"{selected_candidate['cv_mae']:.4f}"
    )

    print(
        "\nBest Parameters:"
    )

    print(
        selected_candidate[
            "best_params"
        ]
    )

    print(
        "\n================================"
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "================================"
    )

    print(
        f"\nMAE: "
        f"{final_metrics['mae']:.4f}"
    )

    print(
        f"RMSE: "
        f"{final_metrics['rmse']:.4f}"
    )

    print(
        f"R²: "
        f"{final_metrics['r2']:.4f}"
    )

    print(
        f"Negative Predictions: "
        f"{final_metrics['negative_prediction_count']}"
    )

    print(
        "\n================================"
    )

    print(
        "FEATURE IMPORTANCE"
    )

    print(
        "================================"
    )

    print(
        "\n"
        + feature_importance.to_string(
            index=False
        )
    )

    print(
        f"\nFinal model saved to:"
        f"\n{model_path}"
    )

    print(
        f"\nMetadata saved to:"
        f"\n{metadata_path}"
    )

    print(
        f"\nFinal-test predictions saved to:"
        f"\n{prediction_path}"
    )


if __name__ == "__main__":

    main()