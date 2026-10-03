import os

import pandas as pd

from src.prediction.recommendation_predictor import (
    RecommendationPredictor
)


def main():

    interaction_path = (
        "artifacts/recommendation/"
        "customer_product_interactions.csv"
    )

    output_directory = (
        "artifacts/recommendation/"
        "predictions"
    )

    output_path = os.path.join(
        output_directory,
        "batch_recommendations.csv"
    )

    # =================================
    # LOAD CUSTOMER IDs
    # =================================

    interactions = pd.read_csv(
        interaction_path
    )

    customer_ids = (
        interactions[
            "CustomerID"
        ]
        .drop_duplicates()
        .head(
            10
        )
        .tolist()
    )

    # =================================
    # PREDICT
    # =================================

    predictor = (
        RecommendationPredictor()
    )

    recommendations = (
        predictor.recommend_batch(
            customer_ids=
                customer_ids,

            top_k=
                5,

            mode=
                "next_purchase"
        )
    )

    # =================================
    # SAVE
    # =================================

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    recommendations.to_csv(
        output_path,
        index=False
    )

    print(
        "\n================================"
    )

    print(
        "BATCH RECOMMENDATIONS"
    )

    print(
        "================================"
    )

    print(
        "\n"
        + recommendations
        .head(
            30
        )
        .to_string(
            index=False
        )
    )

    print(
        f"\nSaved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":

    main()