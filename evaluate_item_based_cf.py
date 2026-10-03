import json
import os

import pandas as pd

from src.recommendation.item_based_cf import (
    ItemBasedCollaborativeRecommender
)

from src.evaluation.recommendation_metrics import (
    RecommendationMetrics
)


def main():

    # =================================
    # PATHS
    # =================================

    interaction_path = (
        "artifacts/recommendation/"
        "recommendation_train_interactions.csv"
    )

    ground_truth_path = (
        "artifacts/recommendation/"
        "recommendation_evaluable_ground_truth.csv"
    )

    output_directory = (
        "artifacts/recommendation/item_cf"
    )

    metrics_path = os.path.join(
        output_directory,
        "item_cf_metrics.json"
    )

    user_results_path = os.path.join(
        output_directory,
        "item_cf_user_results.csv"
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # =================================
    # LOAD DATA
    # =================================

    interactions = pd.read_csv(
        interaction_path
    )

    ground_truth = pd.read_csv(
        ground_truth_path
    )

    # =================================
    # TRAIN ITEM-CF
    # =================================

    recommender = (
        ItemBasedCollaborativeRecommender()
    )

    recommender.fit(
        interactions
    )

    # =================================
    # EVALUATE K=5
    # =================================

    evaluator = (
        RecommendationMetrics()
    )

    result_at_5 = (
        evaluator.evaluate(
            recommender=
                recommender,

            ground_truth=
                ground_truth,

            catalog_products=
                recommender.catalog_products,

            top_k=
                5,

            # Next-purchase prediction
            # allows repeat buying.
            exclude_already_purchased=
                False
        )
    )

    # =================================
    # EVALUATE K=10
    # =================================

    result_at_10 = (
        evaluator.evaluate(
            recommender=
                recommender,

            ground_truth=
                ground_truth,

            catalog_products=
                recommender.catalog_products,

            top_k=
                10,

            exclude_already_purchased=
                False
        )
    )

    # =================================
    # EVALUATE K=20
    # =================================

    result_at_20 = (
        evaluator.evaluate(
            recommender=
                recommender,

            ground_truth=
                ground_truth,

            catalog_products=
                recommender.catalog_products,

            top_k=
                20,

            exclude_already_purchased=
                False
        )
    )

    # =================================
    # BUILD REPORT
    # =================================

    report = {

        "model":
            "Item-Based Collaborative Filtering",

        "similarity":
            "Cosine Similarity",

        "interaction_value":
            "InteractionStrength",

        "evaluation_task":
            "Next-purchase prediction",

        "repeat_purchases_allowed":
            True,

        "metrics": {

            "5":
                result_at_5[
                    "metrics"
                ],

            "10":
                result_at_10[
                    "metrics"
                ],

            "20":
                result_at_20[
                    "metrics"
                ]
        }
    }

    # =================================
    # SAVE
    # =================================

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    result_at_10[
        "user_results"
    ].to_csv(
        user_results_path,
        index=False
    )

    # =================================
    # OUTPUT
    # =================================

    print(
        "\n================================"
    )

    print(
        "ITEM-BASED COLLABORATIVE FILTERING"
    )

    print(
        "================================"
    )

    print(
        f"\nCustomers: "
        f"{len(recommender.customer_ids)}"
    )

    print(
        f"Products: "
        f"{len(recommender.product_ids)}"
    )

    print(
        f"Interactions: "
        f"{recommender.customer_item_matrix.nnz}"
    )

    for k, result in [
        (
            5,
            result_at_5
        ),
        (
            10,
            result_at_10
        ),
        (
            20,
            result_at_20
        )
    ]:

        metrics = (
            result[
                "metrics"
            ]
        )

        print(
            f"\n---------- K={k} ----------"
        )

        print(
            f"Precision@{k}: "
            f"{metrics['precision_at_k']:.4f}"
        )

        print(
            f"Recall@{k}: "
            f"{metrics['recall_at_k']:.4f}"
        )

        print(
            f"HitRate@{k}: "
            f"{metrics['hit_rate_at_k']:.4f}"
        )

        print(
            f"Catalog Coverage: "
            f"{metrics['catalog_coverage']:.4f}"
        )

    print(
        f"\nMetrics saved to:"
        f"\n{metrics_path}"
    )


if __name__ == "__main__":

    main()