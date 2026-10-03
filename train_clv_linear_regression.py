import pandas as pd

from src.preprocessing.clv_preprocessing import (
    CLVPreprocessor
)

from src.training.clv_data_split import (
    CLVDataSplitter
)

from src.training.clv_linear_model import (
    CLVLinearRegressionModel
)

from src.evaluation.clv_evaluation import (
    CLVModelEvaluation
)

from src.analysis.clv_model_analysis import (
    CLVModelAnalysis
)


def main():

    # ---------------------------------
    # Load dataset
    # ---------------------------------

    input_path = (
        "artifacts/clv/"
        "customer_clv_dataset.csv"
    )

    df = pd.read_csv(
        input_path
    )

    # ---------------------------------
    # Preprocess
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
    # Split
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
    # RAW TARGET MODEL
    # =================================

    raw_trainer = (
        CLVLinearRegressionModel()
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
    # LOG TARGET MODEL
    # =================================

    log_trainer = (
        CLVLinearRegressionModel()
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
    # MODEL COMPARISON
    # =================================

    comparison = pd.DataFrame(
        [
            {
                "Model":
                    "Linear Regression",

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
                    "Linear Regression",

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
    # COEFFICIENT ANALYSIS
    # =================================

    analyzer = (
        CLVModelAnalysis()
    )

    raw_coefficients = (
        analyzer
        .analyze_linear_coefficients(
            raw_model,
            X.columns.tolist()
        )
    )

    log_coefficients = (
        analyzer
        .analyze_linear_coefficients(
            log_model,
            X.columns.tolist()
        )
    )

    # =================================
    # CONSOLE OUTPUT
    # =================================

    print(
        "\n================================"
    )

    print(
        "CLV LINEAR REGRESSION"
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
        "RAW TARGET RESULTS"
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
        "\nRaw Model Coefficients:"
    )

    print(
        raw_coefficients.to_string(
            index=False
        )
    )

    print(
        "\n--------------------------------"
    )

    print(
        "LOG TARGET RESULTS"
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
        "\nLog Model Coefficients:"
    )

    print(
        log_coefficients.to_string(
            index=False
        )
    )


if __name__ == "__main__":

    main()