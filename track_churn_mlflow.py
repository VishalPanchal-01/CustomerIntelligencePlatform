from src.mlflow_tracking.churn_tracking import (
    ChurnMLflowTracking
)


def main():

    print(
        "\n"
        "========================================"
    )

    print(
        "CHURN MLFLOW TRACKING"
    )

    print(
        "========================================"
        "\n"
    )

    tracker = (
        ChurnMLflowTracking()
    )

    comparison = (
        tracker.run()
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "CHURN MODEL COMPARISON"
    )

    print(
        "========================================"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "\nArtifacts saved under:"
    )

    print(
        "artifacts/mlflow/churn/"
    )

    print(
        "\nMLflow tracking completed."
    )


if __name__ == "__main__":

    main()