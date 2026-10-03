import pandas as pd

from src.preprocessing.clv_preprocessing import (
    CLVPreprocessor
)

from src.training.clv_data_split import (
    CLVDataSplitter
)

from src.training.clv_gradient_boosting_model import (
    CLVGradientBoostingModel
)

from src.evaluation.clv_evaluation import (
    CLVModelEvaluation
)

from src.analysis.clv_model_analysis import (
    CLVModelAnalysis
)


def main():

    input_path = (
        "artifacts/clv/"
        "customer_clv_dataset.csv"
    )

    # ---------------------------------
    # Load data
    # ---------------------------------

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
    # RAW MODEL
    # =================================

    raw_trainer = (
        CLVGradientBoostingModel()
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
    # LOG MODEL
    # =================================

    log_trainer = (
        CLVGradientBoostingModel()
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
        .analyze_gradient_boosting_importance(
            raw_model,
            X.columns.tolist()
        )
    )

    log_importance = (
        analyzer
        .analyze_gradient_boosting_importance(
            log_model,
            X.columns.tolist()
        )
    )

    # =================================
    # RESULTS
    # =================================

    comparison = pd.DataFrame(
        [
            {
                "Model":
                    "Gradient Boosting",

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
                    ]
            },

            {
                "Model":
                    "Gradient Boosting",

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
                    ]
            }
        ]
    )

    print(
        "\n================================"
    )

    print(
        "CLV GRADIENT BOOSTING"
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
        "\nRAW TARGET FEATURE IMPORTANCE:"
    )

    print(
        raw_importance.to_string(
            index=False
        )
    )

    print(
        "\nLOG TARGET FEATURE IMPORTANCE:"
    )

    print(
        log_importance.to_string(
            index=False
        )
    )


if __name__ == "__main__":

    main()