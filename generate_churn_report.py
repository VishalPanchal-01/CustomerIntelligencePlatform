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

from src.evaluation.churn_evaluation import (
    ChurnModelEvaluation
)

from src.evaluation.final_model_selection import (
    FinalModelSelector
)

from src.evaluation.final_churn_report import (
    FinalChurnReport
)

from src.utils.model_persistence import (
    ModelPersistence
)


def main():

    # ---------------------------------
    # Configuration
    # ---------------------------------

    churn_dataset_path = (
        "artifacts/churn/"
        "customer_churn_dataset.csv"
    )

    model_path = (
        "models/churn/"
        "churn_model.pkl"
    )

    report_path = (
        "artifacts/churn/"
        "churn_model_report.json"
    )

    prediction_days = 90

    # ---------------------------------
    # Load churn dataset
    # ---------------------------------

    churn_df = pd.read_csv(
        churn_dataset_path
    )

    # ---------------------------------
    # Prepare X and y
    # ---------------------------------

    preprocessor = (
        ChurnPreprocessor()
    )

    X, y = (
        preprocessor
        .prepare_features(
            churn_df
        )
    )

    feature_names = (
        X.columns.tolist()
    )

    # ---------------------------------
    # Create final train/test split
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
    # Hyperparameter tuning
    # ---------------------------------

    tuner = (
        ChurnHyperparameterTuner()
    )

    logistic_search = (
        tuner
        .tune_logistic_regression(
            X_train,
            y_train
        )
    )

    random_forest_search = (
        tuner
        .tune_random_forest(
            X_train,
            y_train
        )
    )

    # ---------------------------------
    # Tuned candidate models
    # ---------------------------------

    models = {
        "Logistic Regression":
            logistic_search.best_estimator_,

        "Random Forest":
            random_forest_search.best_estimator_
    }

    # ---------------------------------
    # Select final model
    # ---------------------------------

    selector = (
        FinalModelSelector()
    )

    (
        best_model_name,
        best_model,
        comparison
    ) = selector.select_best_model(
        models,
        X_test,
        y_test
    )

    # ---------------------------------
    # Evaluate selected model
    # ---------------------------------

    evaluator = (
        ChurnModelEvaluation()
    )

    final_evaluation = (
        evaluator.evaluate(
            best_model,
            X_test,
            y_test
        )
    )

    # ---------------------------------
    # Save selected model
    # ---------------------------------

    persistence = (
        ModelPersistence()
    )

    persistence.save_model(
        best_model,
        model_path
    )

    # ---------------------------------
    # Generate final report
    # ---------------------------------

    report_generator = (
        FinalChurnReport()
    )

    report = (
        report_generator
        .generate_report(
            model_name=best_model_name,
            model=best_model,
            evaluation_results=
                final_evaluation,
            churn_df=churn_df,
            feature_names=
                feature_names,
            model_path=model_path,
            prediction_days=
                prediction_days,
            output_path=
                report_path
        )
    )

    # ---------------------------------
    # Console output
    # ---------------------------------

    print(
        "\n================================"
    )

    print(
        "FINAL CHURN MODEL REPORT"
    )

    print(
        "================================"
    )

    print(
        f"\nSelected Model: "
        f"{best_model_name}"
    )

    print(
        f"\nCustomers: "
        f"{report['dataset']['customers']}"
    )

    print(
        "\nFeatures:"
    )

    for feature in (
        report["features"]
    ):

        print(
            f"- {feature}"
        )

    print(
        "\nTest Metrics:"
    )

    print(
        f"Accuracy: "
        f"{final_evaluation['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{final_evaluation['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{final_evaluation['recall']:.4f}"
    )

    print(
        f"F1 Score: "
        f"{final_evaluation['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{final_evaluation['roc_auc']:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        final_evaluation[
            "confusion_matrix"
        ]
    )

    print(
        "\nModel Comparison:"
    )

    display_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1Score",
        "ROCAUC"
    ]

    print(
        comparison[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        f"\nModel saved to:"
        f"\n{model_path}"
    )

    print(
        f"\nReport saved to:"
        f"\n{report_path}"
    )


if __name__ == "__main__":

    main()