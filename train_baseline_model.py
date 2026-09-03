import pandas as pd

from src.preprocessing.churn_preprocessing import (
    ChurnPreprocessor
)

from src.training.data_split import (
    ChurnDataSplitter
)

from src.training.baseline_model import (
    BaselineChurnModel
)

from src.evaluation.churn_evaluation import (
    ChurnModelEvaluation
)


def main():

    # -------------------------
    # Load dataset
    # -------------------------

    df = pd.read_csv(
        "artifacts/churn/customer_churn_dataset.csv"
    )

    # -------------------------
    # Prepare X and y
    # -------------------------

    preprocessor = (
        ChurnPreprocessor()
    )

    X, y = (
        preprocessor
        .prepare_features(df)
    )

    # -------------------------
    # Train-test split
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
    # Train baseline
    # -------------------------

    trainer = (
        BaselineChurnModel()
    )

    model = trainer.train(
        X_train,
        y_train
    )

    # -------------------------
    # Evaluate
    # -------------------------

    evaluator = (
        ChurnModelEvaluation()
    )

    results = evaluator.evaluate(
        model,
        X_test,
        y_test
    )

    # -------------------------
    # Results
    # -------------------------

    print(
        "\nBaseline Model Results:"
    )

    print(
        f"Accuracy: "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{results['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{results['recall']:.4f}"
    )

    print(
        f"F1 Score: "
        f"{results['f1_score']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        results[
            "confusion_matrix"
        ]
    )


if __name__ == "__main__":

    main()