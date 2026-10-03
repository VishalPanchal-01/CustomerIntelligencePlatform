import os

import pandas as pd

from src.preprocessing.clv_preprocessing import (
    CLVPreprocessor
)

from src.training.clv_data_split import (
    CLVDataSplitter
)

from src.evaluation.clv_cross_validation import (
    CLVCrossValidator
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
        "clv_cross_validation.csv"
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

    # =================================
    # FINAL TRAIN/TEST SPLIT
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
    # IMPORTANT:
    # X_test / y_raw_test remain
    # completely untouched here.
    # ---------------------------------

    # =================================
    # CROSS-VALIDATE TRAINING SET
    # =================================

    validator = (
        CLVCrossValidator(
            n_splits=5,
            random_state=42
        )
    )

    cv_results = (
        validator.evaluate_models(
            X=
                X_train,

            y_raw=
                y_raw_train,

            y_log=
                y_log_train
        )
    )

    # ---------------------------------
    # Save report
    # ---------------------------------

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    cv_results.to_csv(
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
        "CLV CROSS-VALIDATION RESULTS"
    )

    print(
        "================================"
    )

    print(
        "\n"
        + cv_results.to_string(
            index=False
        )
    )

    print(
        "\n================================"
    )

    print(
        "FINAL TEST SET STATUS"
    )

    print(
        "================================"
    )

    print(
        f"\nTraining customers: "
        f"{len(X_train)}"
    )

    print(
        f"Final test customers: "
        f"{len(X_test)}"
    )

    print(
        "\nFinal test set was NOT used "
        "during cross-validation."
    )

    print(
        f"\nCross-validation report saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":

    main()