import os

import pandas as pd

from src.preprocessing.clv_preprocessing import (
    CLVPreprocessor
)

from src.training.clv_data_split import (
    CLVDataSplitter
)

from src.evaluation.clv_model_comparison import (
    CLVModelComparison
)


def main():

    # ---------------------------------
    # Paths
    # ---------------------------------

    input_path = (
        "artifacts/clv/"
        "customer_clv_dataset.csv"
    )

    output_directory = (
        "artifacts/clv"
    )

    output_path = os.path.join(
        output_directory,
        "clv_model_comparison.csv"
    )

    # ---------------------------------
    # Load CLV dataset
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

    # ---------------------------------
    # Compare models
    # ---------------------------------

    comparator = (
        CLVModelComparison()
    )

    comparison = (
        comparator.compare_models(
            X_train=
                X_train,

            X_test=
                X_test,

            y_raw_train=
                y_raw_train,

            y_raw_test=
                y_raw_test,

            y_log_train=
                y_log_train
        )
    )

    # ---------------------------------
    # Save comparison
    # ---------------------------------

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    comparison.to_csv(
        output_path,
        index=False
    )

    # ---------------------------------
    # Console output
    # ---------------------------------

    print(
        "\n================================"
    )

    print(
        "CLV MODEL COMPARISON"
    )

    print(
        "================================"
    )

    print(
        "\n"
        + comparison.to_string(
            index=False
        )
    )

    print(
        f"\nComparison saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":

    main()