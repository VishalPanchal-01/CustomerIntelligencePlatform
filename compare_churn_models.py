import pandas as pd

from src.preprocessing.churn_preprocessing import (
    ChurnPreprocessor
)

from src.training.data_split import (
    ChurnDataSplitter
)

from src.evaluation.model_comparison import (
    ChurnModelComparison
)


def main():

    # -------------------------
    # Load dataset
    # -------------------------

    df = pd.read_csv(
        "artifacts/churn/customer_churn_dataset.csv"
    )

    # -------------------------
    # Prepare features
    # -------------------------

    preprocessor = (
        ChurnPreprocessor()
    )

    X, y = (
        preprocessor
        .prepare_features(df)
    )

    # -------------------------
    # Split dataset once
    # -------------------------

    splitter = (
        ChurnDataSplitter()
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = splitter.split_data(
        X,
        y
    )

    # -------------------------
    # Compare models
    # -------------------------

    comparison = (
        ChurnModelComparison()
    )

    results = (
        comparison.compare_models(
            X_train,
            X_test,
            y_train,
            y_test
        )
    )

    # -------------------------
    # Display
    # -------------------------

    print(
        "\nChurn Model Comparison:"
    )

    print(
        results.to_string(
            index=False
        )
    )


if __name__ == "__main__":

    main()