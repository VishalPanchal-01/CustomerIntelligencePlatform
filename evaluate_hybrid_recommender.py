import json
import os

import pandas as pd

from src.recommendation.hybrid_recommender import (
    HybridRecommender
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
        "artifacts/recommendation/"
        "hybrid"
    )

    metrics_path = os.path.join(
        output_directory,
        "hybrid_metrics.json"
    )

    user_results_path = os.path.join(
        output_directory,
        "hybrid_user_results.csv"
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # =================================
    # LOAD
    # =================================

    interactions = pd.read_csv(
        interaction_path
    )

    ground_truth = pd.read_csv(
        ground_truth_path
    )

    # =================================
    # TRAIN HYBRID
    # =================================

    recommender = (
        HybridRecommender(
            item_cf_weight=0.40,
            matrix_factorization_weight=0.40,
            popularity_weight=0.20,
            n_components=20,
            random_state=42,
            candidate_multiplier=5
        )
    )

    recommender.fit(
        interactions
    )

    # =================================
    # EVALUATE
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

            exclude_already_purchased=
                False
        )
    )

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
    # REPORT
    # =================================

    report = {

        "model":
            "Hybrid Recommender",

        "components": [
            "Item-Based Collaborative Filtering",
            "Matrix Factorization",
            "Popularity"
        ],

        "weights": {

            "item_cf":
                recommender.item_cf_weight,

            "matrix_factorization":
                recommender.matrix_factorization_weight,

            "popularity":
                recommender.popularity_weight
        },

        "score_normalization":
            "Per-component Min-Max normalization",

        "candidate_multiplier":
            recommender.candidate_multiplier,

        "cold_start_fallback":
            "Popularity",

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
        "HYBRID RECOMMENDATION SYSTEM"
    )

    print(
        "================================"
    )

    print(
        f"\nItem-CF Weight: "
        f"{recommender.item_cf_weight:.2f}"
    )

    print(
        f"Matrix Factorization Weight: "
        f"{recommender.matrix_factorization_weight:.2f}"
    )

    print(
        f"Popularity Weight: "
        f"{recommender.popularity_weight:.2f}"
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