import numpy as np
import pandas as pd

from src.preprocessing.clv_preprocessing import (
    CLVPreprocessor
)

from src.training.clv_data_split import (
    CLVDataSplitter
)


def main():

    # ---------------------------------
    # Dataset path
    # ---------------------------------

    input_path = (
        "artifacts/clv/"
        "customer_clv_dataset.csv"
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

    # ---------------------------------
    # Display dataset information
    # ---------------------------------

    print(
        "\n================================"
    )

    print(
        "CLV TRAINING DATA PREPARATION"
    )

    print(
        "================================"
    )

    print(
        "\nFeatures:"
    )

    print(
        X.columns.tolist()
    )

    print(
        "\nFull Dataset Shape:"
    )

    print(
        X.shape
    )

    print(
        "\nTraining Shape:"
    )

    print(
        X_train.shape
    )

    print(
        "\nTesting Shape:"
    )

    print(
        X_test.shape
    )

    # ---------------------------------
    # Raw target information
    # ---------------------------------

    print(
        "\n================================"
    )

    print(
        "RAW TARGET"
    )

    print(
        "================================"
    )

    print(
        "\nTraining Target Statistics:"
    )

    print(
        y_raw_train.describe()
    )

    print(
        "\nTesting Target Statistics:"
    )

    print(
        y_raw_test.describe()
    )

    # ---------------------------------
    # Log target information
    # ---------------------------------

    print(
        "\n================================"
    )

    print(
        "LOG TARGET"
    )

    print(
        "================================"
    )

    print(
        "\nTraining Log Target Statistics:"
    )

    print(
        y_log_train.describe()
    )

    print(
        "\nTesting Log Target Statistics:"
    )

    print(
        y_log_test.describe()
    )

    # ---------------------------------
    # Compare skewness
    # ---------------------------------

    print(
        "\n================================"
    )

    print(
        "TARGET SKEWNESS COMPARISON"
    )

    print(
        "================================"
    )

    print(
        f"\nRaw FutureRevenue Skewness: "
        f"{y_raw_train.skew():.4f}"
    )

    print(
        f"Log FutureRevenue Skewness: "
        f"{y_log_train.skew():.4f}"
    )

    # ---------------------------------
    # Verify inverse transformation
    # ---------------------------------

    sample_log_value = (
        y_log_train.iloc[0]
    )

    restored_value = (
        np.expm1(
            sample_log_value
        )
    )

    original_value = (
        y_raw_train.iloc[0]
    )

    print(
        "\nInverse Transformation Check:"
    )

    print(
        f"Original Revenue: "
        f"{original_value:.4f}"
    )

    print(
        f"Restored Revenue: "
        f"{restored_value:.4f}"
    )


if __name__ == "__main__":

    main()