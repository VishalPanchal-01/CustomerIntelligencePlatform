import pandas as pd

from src.preprocessing.churn_preprocessing import (
    ChurnPreprocessor
)

from src.training.data_split import (
    ChurnDataSplitter
)

from src.evaluation.cross_validation import (
    ChurnCrossValidation
)


def main():

    df = pd.read_csv(
        "artifacts/churn/customer_churn_dataset.csv"
    )

    preprocessor = (
        ChurnPreprocessor()
    )

    X, y = (
        preprocessor
        .prepare_features(df)
    )

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

    validator = (
        ChurnCrossValidation()
    )

    results = (
        validator.evaluate_models(
            X_train,
            y_train,
            n_splits=5
        )
    )

    print(
        "\nCross-Validation Results:"
    )

    print(
        results.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()