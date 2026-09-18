import pandas as pd

from src.preprocessing.churn_preprocessing import (
    ChurnPreprocessor
)

from src.training.data_split import (
    ChurnDataSplitter
)

from src.training.hyperparameter_tuning import (
    ChurnHyperparameterTuner
)


def main():

    # ---------------------------------
    # Load dataset
    # ---------------------------------

    df = pd.read_csv(
        "artifacts/churn/customer_churn_dataset.csv"
    )

    # ---------------------------------
    # Prepare X and y
    # ---------------------------------

    preprocessor = (
        ChurnPreprocessor()
    )

    X, y = (
        preprocessor
        .prepare_features(df)
    )

    # ---------------------------------
    # Split data
    # ---------------------------------

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

    # ---------------------------------
    # Initialize tuner
    # ---------------------------------

    tuner = (
        ChurnHyperparameterTuner()
    )

    # ---------------------------------
    # Tune Logistic Regression
    # ---------------------------------

    logistic_search = (
        tuner
        .tune_logistic_regression(
            X_train,
            y_train
        )
    )

    print(
        "\nBest Logistic Regression Parameters:"
    )

    print(
        logistic_search.best_params_
    )

    print(
        "\nBest Logistic Regression CV F1:"
    )

    print(
        logistic_search.best_score_
    )

    # ---------------------------------
    # Tune Random Forest
    # ---------------------------------

    random_forest_search = (
        tuner
        .tune_random_forest(
            X_train,
            y_train
        )
    )

    print(
        "\nBest Random Forest Parameters:"
    )

    print(
        random_forest_search.best_params_
    )

    print(
        "\nBest Random Forest CV F1:"
    )

    print(
        random_forest_search.best_score_
    )


if __name__ == "__main__":

    main()