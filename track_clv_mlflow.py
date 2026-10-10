from src.mlflow_tracking.clv_tracking import (
    CLVMLflowTracking,
)


def main():

    print(
        "\n"
        "========================================"
    )

    print(
        "CLV MLFLOW TRACKING"
    )

    print(
        "========================================"
        "\n"
    )

    tracker = (
        CLVMLflowTracking()
    )

    comparison = (
        tracker.run()
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "CLV MODEL COMPARISON"
    )

    print(
        "========================================"
    )

    if comparison.empty:

        print(
            "No CLV model results generated."
        )

        return

    columns_to_show = [
        column
        for column in [
            "model",
            "cv_mae_mean",
            "cv_rmse_mean",
            "cv_r2_mean",
            "test_mae",
            "test_rmse",
            "test_r2",
            "run_id",
        ]
        if column in comparison.columns
    ]

    print(
        comparison[
            columns_to_show
        ].to_string(
            index=False
        )
    )

    print(
        "\nArtifacts saved under:"
    )

    print(
        "artifacts/mlflow/clv/"
    )

    print(
        "\nMLflow experiment:"
    )

    print(
        "90-Day Revenue Prediction"
    )

    print(
        "\nCLV MLflow tracking completed."
    )


if __name__ == "__main__":

    main()