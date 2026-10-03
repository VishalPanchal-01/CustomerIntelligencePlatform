import json
import os

import pandas as pd

from src.recommendation.popularity_recommender import (
    PopularityRecommender
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
        "artifacts/recommendation/baseline"
    )

    ranking_path = os.path.join(
        output_directory,
        "popularity_product_ranking.csv"
    )

    metrics_path = os.path.join(
        output_directory,
        "popularity_metrics.json"
    )

    user_results_path = os.path.join(
        output_directory,
        "popularity_user_results.csv"
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
    # TRAIN POPULARITY MODEL
    # =================================

    recommender = (
        PopularityRecommender()
    )

    recommender.fit(
        interactions
    )

    # =================================
    # SAVE PRODUCT RANKING
    # =================================

    recommender.product_ranking.to_csv(
        ranking_path,
        index=False
    )

    # =================================
    # EVALUATE AT K = 5
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

            # Next-purchase evaluation:
            # repeat purchases are allowed.
            exclude_already_purchased=
                False
        )
    )

    # =================================
    # EVALUATE AT K = 10
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
    # EVALUATE AT K = 20
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
    # BUILD METRIC REPORT
    # =================================

    metrics_report = {

        "model":
            "Popularity Baseline",

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
    # SAVE METRICS
    # =================================

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics_report,
            file,
            indent=4
        )

    # ---------------------------------
    # Save K=10 user-level results
    # ---------------------------------

    result_at_10[
        "user_results"
    ].to_csv(
        user_results_path,
        index=False
    )

    # =================================
    # CONSOLE OUTPUT
    # =================================

    print(
        "\n================================"
    )

    print(
        "POPULARITY RECOMMENDER BASELINE"
    )

    print(
        "================================"
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
        "\n================================"
    )

    print(
        "TOP 10 POPULAR PRODUCTS"
    )

    print(
        "================================"
    )

    print(
        "\n"
        + recommender.product_ranking[
            [
                "StockCode",
                "Description",
                "CustomerCount",
                "PurchaseCount",
                "TotalQuantity",
                "PopularityScore"
            ]
        ]
        .head(
            10
        )
        .to_string(
            index=False
        )
    )

    print(
        f"\nMetrics saved to:"
        f"\n{metrics_path}"
    )


if __name__ == "__main__":

    main()