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


def main():

    # ---------------------------------
    # Configuration
    # ---------------------------------

    input_path = (
        "artifacts/clv/"
        "customer_clv_dataset.csv"
    )

    output_directory = (
        "artifacts/clv/tuning"
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # ---------------------------------
    # Load dataset
    # ---------------------------------

    df = pd.read_csv(
        input_path
    )

    # ---------------------------------
    # Prepare features and targets
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

    # ---------------------------------
    # Important:
    # X_test and y_raw_test are not
    # used anywhere below.
    # ---------------------------------

    tuner = (
        CLVHyperparameterTuner(
            n_splits=5,
            random_state=42
        )
    )

    # =================================
    # RANDOM FOREST RAW
    # =================================

    rf_raw_search = (
        tuner.tune_random_forest_raw(
            X_train=
                X_train,

            y_train=
                y_raw_train,

            n_iter=15
        )
    )

    # =================================
    # RANDOM FOREST LOG
    # =================================

    rf_log_search = (
        tuner.tune_random_forest_log(
            X_train=
                X_train,

            y_train_raw=
                y_raw_train,

            n_iter=15
        )
    )

    # =================================
    # GRADIENT BOOSTING RAW
    # =================================

    gb_raw_search = (
        tuner.tune_gradient_boosting_raw(
            X_train=
                X_train,

            y_train=
                y_raw_train,

            n_iter=15
        )
    )

    # =================================
    # GRADIENT BOOSTING LOG
    # =================================

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
    # COMPARISON
    # =================================

    tuning_comparison = pd.DataFrame(
        [
            {
                "Model":
                    "Random Forest",

                "Target":
                    "Raw",

                "BestCVMAE":
                    -rf_raw_search.best_score_
            },

            {
                "Model":
                    "Random Forest",

                "Target":
                    "Log",

                "BestCVMAE":
                    -rf_log_search.best_score_
            },

            {
                "Model":
                    "Gradient Boosting",

                "Target":
                    "Raw",

                "BestCVMAE":
                    -gb_raw_search.best_score_
            },

            {
                "Model":
                    "Gradient Boosting",

                "Target":
                    "Log",

                "BestCVMAE":
                    -gb_log_search.best_score_
            }
        ]
    )

    tuning_comparison = (
        tuning_comparison
        .sort_values(
            by="BestCVMAE",
            ascending=True
        )
        .reset_index(
            drop=True
        )
    )

    # =================================
    # SAVE COMPARISON
    # =================================

    comparison_path = os.path.join(
        output_directory,
        "clv_tuning_comparison.csv"
    )

    tuning_comparison.to_csv(
        comparison_path,
        index=False
    )

    # =================================
    # SAVE BEST PARAMETER REPORT
    # =================================

    best_parameters = {

        "random_forest_raw": {
            "best_cv_mae":
                float(
                    -rf_raw_search.best_score_
                ),

            "best_params":
                rf_raw_search.best_params_
        },

        "random_forest_log": {
            "best_cv_mae":
                float(
                    -rf_log_search.best_score_
                ),

            "best_params":
                rf_log_search.best_params_
        },

        "gradient_boosting_raw": {
            "best_cv_mae":
                float(
                    -gb_raw_search.best_score_
                ),

            "best_params":
                gb_raw_search.best_params_
        },

        "gradient_boosting_log": {
            "best_cv_mae":
                float(
                    -gb_log_search.best_score_
                ),

            "best_params":
                gb_log_search.best_params_
        }
    }

    parameter_path = os.path.join(
        output_directory,
        "clv_best_parameters.json"
    )

    with open(
        parameter_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            best_parameters,
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
        "CLV HYPERPARAMETER TUNING"
    )

    print(
        "================================"
    )

    print(
        "\nTuning Comparison:"
    )

    print(
        tuning_comparison.to_string(
            index=False
        )
    )

    print(
        "\n--------------------------------"
    )

    print(
        "RANDOM FOREST RAW"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Best CV MAE: "
        f"{-rf_raw_search.best_score_:.4f}"
    )

    print(
        "Best Parameters:"
    )

    print(
        rf_raw_search.best_params_
    )

    print(
        "\n--------------------------------"
    )

    print(
        "RANDOM FOREST LOG"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Best CV MAE: "
        f"{-rf_log_search.best_score_:.4f}"
    )

    print(
        "Best Parameters:"
    )

    print(
        rf_log_search.best_params_
    )

    print(
        "\n--------------------------------"
    )

    print(
        "GRADIENT BOOSTING RAW"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Best CV MAE: "
        f"{-gb_raw_search.best_score_:.4f}"
    )

    print(
        "Best Parameters:"
    )

    print(
        gb_raw_search.best_params_
    )

    print(
        "\n--------------------------------"
    )

    print(
        "GRADIENT BOOSTING LOG"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Best CV MAE: "
        f"{-gb_log_search.best_score_:.4f}"
    )

    print(
        "Best Parameters:"
    )

    print(
        gb_log_search.best_params_
    )

    print(
        "\n================================"
    )

    print(
        "FINAL TEST SET"
    )

    print(
        "================================"
    )

    print(
        f"\nFinal test customers: "
        f"{len(X_test)}"
    )

    print(
        "\nThe final test set was NOT "
        "used during hyperparameter tuning."
    )

    print(
        f"\nTuning comparison saved to:"
        f"\n{comparison_path}"
    )

    print(
        f"\nBest parameters saved to:"
        f"\n{parameter_path}"
    )


if __name__ == "__main__":

    main()