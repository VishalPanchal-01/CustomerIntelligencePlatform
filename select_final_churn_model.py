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

from src.evaluation.final_model_selection import (
    FinalModelSelector
)

from src.utils.model_persistence import (
    ModelPersistence
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
    # Train/Test split
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
    # Tune models
    # -------------------------

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

    # -------------------------
    # Best estimators
    # -------------------------

    tuned_models = {
        "Logistic Regression":
            logistic_search.best_estimator_,

        "Random Forest":
            random_forest_search.best_estimator_
    }

    # -------------------------
    # Final selection
    # -------------------------

    selector = (
        FinalModelSelector()
    )

    (
        best_model_name,
        best_model,
        comparison
    ) = selector.select_best_model(
        tuned_models,
        X_test,
        y_test
    )

    # -------------------------
    # Print comparison
    # -------------------------

    print(
        "\nFinal Test Set Comparison:"
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
        "\nSelected Final Model:"
    )

    print(
        best_model_name
    )

    print(
        "\nBest Model Confusion Matrix:"
    )

    print(
        comparison.iloc[0][
            "ConfusionMatrix"
        ]
    )

    # -------------------------
# Save final model
# -------------------------

    persistence = (
        ModelPersistence()
        )

    model_path = (
        "models/churn/churn_model.pkl"
    )

    persistence.save_model(best_model,model_path)

    print(f"\nFinal model saved to: "f"{model_path}")


if __name__ == "__main__":

    main()