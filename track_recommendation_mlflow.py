from src.mlflow_tracking.recommendation_tracking import (
    RecommendationMLflowTracking,
)


def main():

    print(
        "\n"
        "========================================"
    )

    print(
        "RECOMMENDATION MLFLOW TRACKING"
    )

    print(
        "========================================"
        "\n"
    )

    tracker = (
        RecommendationMLflowTracking()
    )

    results = (
        tracker.run()
    )

    print(
        "\n"
        "========================================"
    )

    print(
        "RECOMMENDATION MODEL COMPARISON"
    )

    print(
        "========================================"
    )

    if results.empty:

        print(
            "No recommendation runs "
            "were generated."
        )

        return

    display_columns = [
        column
        for column in [
            "model",
            "selected",
            "top_k",
            "precision_at_k",
            "recall_at_k",
            "hit_rate_at_k",
            "catalog_coverage",
            "run_id",
        ]
        if column in results.columns
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nArtifacts saved under:"
    )

    print(
        "artifacts/mlflow/recommendation/"
    )

    print(
        "\nMLflow experiment:"
    )

    print(
        "Next Product Recommendation"
    )

    print(
        "\nRecommendation MLflow "
        "tracking completed."
    )


if __name__ == "__main__":

    main()
    