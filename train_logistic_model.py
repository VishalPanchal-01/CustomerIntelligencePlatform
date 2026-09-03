import pandas as pd

from src.preprocessing.churn_preprocessing import (
    ChurnPreprocessor
)

from src.training.data_split import (
    ChurnDataSplitter
)

from src.training.logistic_model import (
    LogisticChurnModel
)

from src.evaluation.churn_evaluation import (
    ChurnModelEvaluation
)

from src.analysis.model_analysis import (
    ModelAnalysis
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
    # Split data
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
    # Train model
    # -------------------------

    trainer = (
        LogisticChurnModel()
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
    # Print
    # -------------------------

    print(
        "\nLogistic Regression Results:"
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
        f"ROC-AUC: "
        f"{results['roc_auc']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        results[
            "confusion_matrix"
        ]
    )


    analysis = ModelAnalysis()

    coefficient_report = (analysis.analyze_logistic_coefficients(model,X_train.columns))

    print("\nLogistic Regression Coefficients:")

    print(coefficient_report)


if __name__ == "__main__":

    main()