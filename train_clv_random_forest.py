import pandas as pd

from src.preprocessing.clv_preprocessing import (
    CLVPreprocessor
)

from src.training.clv_data_split import (
    CLVDataSplitter
)

from src.training.clv_random_forest_model import (
    CLVRandomForestModel
)

from src.evaluation.clv_evaluation import (
    CLVModelEvaluation
)

from src.analysis.clv_model_analysis import (
    CLVModelAnalysis
)


def main():

    # ---------------------------------
    # Load CLV dataset
    # ---------------------------------

    input_path = (
        "artifacts/clv/"
        "customer_clv_dataset.csv"
    )

    df = pd.read_csv(
        input_path
    )

    # ---------------------------------
    # Prepare features
    # ---------------------------------

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

    # ---------------------------------
    # Split data
    # ---------------------------------

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
    # RAW TARGET RANDOM FOREST
    # =================================

    raw_trainer = (
        CLVRandomForestModel(
            n_estimators=200,
            random_state=42
        )
    )

    raw_model = (
        raw_trainer.train(
            X_train,
            y_raw_train
        )
    )

    evaluator = (
        CLVModelEvaluation()
    )

    raw_results = (
        evaluator.evaluate(
            model=
                raw_model,

            X_test=
                X_test,

            y_test_raw=
                y_raw_test,

            target_type=
                "raw"
        )
    )

    # =================================
    # LOG TARGET RANDOM FOREST
    # =================================

    log_trainer = (
        CLVRandomForestModel(
            n_estimators=200,
            random_state=42
        )
    )

    log_model = (
        log_trainer.train(
            X_train,
            y_log_train
        )
    )

    log_results = (
        evaluator.evaluate(
            model=
                log_model,

            X_test=
                X_test,

            y_test_raw=
                y_raw_test,

            target_type=
                "log"
        )
    )

    # =================================
    # FEATURE IMPORTANCE
    # =================================

    analyzer = (
        CLVModelAnalysis()
    )

    raw_importance = (
        analyzer
        .analyze_random_forest_importance(
            raw_model,
            X.columns.tolist()
        )
    )

    log_importance = (
        analyzer
        .analyze_random_forest_importance(
            log_model,
            X.columns.tolist()
        )
    )

    # =================================
    # MODEL COMPARISON
    # =================================

    comparison = pd.DataFrame(
        [
            {
                "Model":
                    "Random Forest",

                "Target":
                    "Raw",

                "MAE":
                    raw_results[
                        "mae"
                    ],

                "RMSE":
                    raw_results[
                        "rmse"
                    ],

                "R2":
                    raw_results[
                        "r2"
                    ],

                "NegativePredictions":
                    raw_results[
                        "negative_prediction_count"
                    ]
            },

            {
                "Model":
                    "Random Forest",

                "Target":
                    "Log",

                "MAE":
                    log_results[
                        "mae"
                    ],

                "RMSE":
                    log_results[
                        "rmse"
                    ],

                "R2":
                    log_results[
                        "r2"
                    ],

                "NegativePredictions":
                    log_results[
                        "negative_prediction_count"
                    ]
            }
        ]
    )

    # =================================
    # CONSOLE OUTPUT
    # =================================

    print(
        "\n================================"
    )

    print(
        "CLV RANDOM FOREST REGRESSION"
    )

    print(
        "================================"
    )

    print(
        "\nModel Comparison:"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\n--------------------------------"
    )

    print(
        "RAW TARGET RANDOM FOREST"
    )

    print(
        "--------------------------------"
    )

    print(
        f"MAE: "
        f"{raw_results['mae']:.4f}"
    )

    print(
        f"RMSE: "
        f"{raw_results['rmse']:.4f}"
    )

    print(
        f"R²: "
        f"{raw_results['r2']:.4f}"
    )

    print(
        f"Negative Predictions: "
        f"{raw_results['negative_prediction_count']}"
    )

    print(
        "\nRaw Target Feature Importance:"
    )

    print(
        raw_importance.to_string(
            index=False
        )
    )

    print(
        "\n--------------------------------"
    )

    print(
        "LOG TARGET RANDOM FOREST"
    )

    print(
        "--------------------------------"
    )

    print(
        f"MAE: "
        f"{log_results['mae']:.4f}"
    )

    print(
        f"RMSE: "
        f"{log_results['rmse']:.4f}"
    )

    print(
        f"R²: "
        f"{log_results['r2']:.4f}"
    )

    print(
        f"Negative Predictions: "
        f"{log_results['negative_prediction_count']}"
    )

    print(
        "\nLog Target Feature Importance:"
    )

    print(
        log_importance.to_string(
            index=False
        )
    )


if __name__ == "__main__":

    main()