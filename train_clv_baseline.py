import pandas as pd

from src.preprocessing.clv_preprocessing import (
    CLVPreprocessor
)

from src.training.clv_data_split import (
    CLVDataSplitter
)

from src.training.clv_baseline_model import (
    CLVBaselineModel
)

from src.evaluation.clv_evaluation import (
    CLVModelEvaluation
)


def main():

    # ---------------------------------
    # Dataset
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
    # Train/test split
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
    # RAW TARGET BASELINE
    # =================================

    raw_trainer = (
        CLVBaselineModel(
            strategy="median"
        )
    )

    raw_model = raw_trainer.train(
        X_train,
        y_raw_train
    )

    evaluator = (
        CLVModelEvaluation()
    )

    raw_results = (
        evaluator.evaluate(
            model=raw_model,
            X_test=X_test,
            y_test_raw=y_raw_test,
            target_type="raw"
        )
    )

    # =================================
    # LOG TARGET BASELINE
    # =================================

    log_trainer = (
        CLVBaselineModel(
            strategy="median"
        )
    )

    log_model = log_trainer.train(
        X_train,
        y_log_train
    )

    log_results = (
        evaluator.evaluate(
            model=log_model,
            X_test=X_test,
            y_test_raw=y_raw_test,
            target_type="log"
        )
    )

    # =================================
    # COMPARISON
    # =================================

    comparison = pd.DataFrame(
        [
            {
                "Model":
                    "Dummy Median",

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
                    "Dummy Median",

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

    # ---------------------------------
    # Console output
    # ---------------------------------

    print(
        "\n================================"
    )

    print(
        "CLV BASELINE MODEL COMPARISON"
    )

    print(
        "================================"
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
        "RAW TARGET BASELINE"
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
        f"Actual Mean: "
        f"{raw_results['actual_mean']:.4f}"
    )

    print(
        f"Predicted Mean: "
        f"{raw_results['predicted_mean']:.4f}"
    )

    print(
        f"Actual Median: "
        f"{raw_results['actual_median']:.4f}"
    )

    print(
        f"Predicted Median: "
        f"{raw_results['predicted_median']:.4f}"
    )

    print(
        "\n--------------------------------"
    )

    print(
        "LOG TARGET BASELINE"
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
        f"Actual Mean: "
        f"{log_results['actual_mean']:.4f}"
    )

    print(
        f"Predicted Mean: "
        f"{log_results['predicted_mean']:.4f}"
    )

    print(
        f"Actual Median: "
        f"{log_results['actual_median']:.4f}"
    )

    print(
        f"Predicted Median: "
        f"{log_results['predicted_median']:.4f}"
    )


if __name__ == "__main__":

    main()