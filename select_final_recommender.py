import json
import os

import pandas as pd

from src.evaluation.final_recommendation_selection import (
    FinalRecommendationSelector
)


def make_json_serializable(
    value
):

    if isinstance(
        value,
        dict
    ):

        return {
            str(key):
                make_json_serializable(
                    item
                )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        list
    ):

        return [
            make_json_serializable(
                item
            )
            for item in value
        ]

    if hasattr(
        value,
        "item"
    ):

        return value.item()

    return value


def main():

    # =================================
    # PATHS
    # =================================

    comparison_path = (
        "artifacts/recommendation/"
        "recommendation_model_comparison.csv"
    )

    output_directory = (
        "artifacts/recommendation/final"
    )

    selection_path = os.path.join(
        output_directory,
        "recommendation_final_selection.json"
    )

    ranked_path = os.path.join(
        output_directory,
        "recommendation_final_comparison.csv"
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # =================================
    # LOAD MODEL COMPARISON
    # =================================

    if not os.path.exists(
        comparison_path
    ):

        raise FileNotFoundError(
            f"Recommendation comparison file "
            f"not found: {comparison_path}"
        )

    comparison = pd.read_csv(
        comparison_path
    )

    # =================================
    # SELECT FINAL STRATEGY
    # =================================

    selector = (
        FinalRecommendationSelector(
            selection_k=10
        )
    )

    selection = (
        selector.select_model(
            comparison
        )
    )

    # =================================
    # SAVE RANKED K=10 COMPARISON
    # =================================

    ranked_models = pd.DataFrame(
        selection[
            "ranked_models"
        ]
    )

    ranked_models.to_csv(
        ranked_path,
        index=False
    )

    # =================================
    # DETERMINE MODEL CONFIGURATION
    # =================================

    selected_model = (
        selection[
            "selected_model"
        ]
    )

    model_configuration = {}

    if (
        selected_model
        ==
        "Popularity Baseline"
    ):

        model_configuration = {
            "type":
                "PopularityRecommender",

            "personalized":
                False,

            "cold_start_supported":
                True
        }

    elif (
        selected_model
        ==
        "Item-Based Collaborative Filtering"
    ):

        model_configuration = {
            "type":
                "ItemBasedCollaborativeRecommender",

            "similarity":
                "Cosine Similarity",

            "interaction_value":
                "InteractionStrength",

            "personalized":
                True,

            "cold_start_supported":
                False
        }

    elif (
        selected_model
        ==
        "Matrix Factorization"
    ):

        model_configuration = {
            "type":
                "MatrixFactorizationRecommender",

            "algorithm":
                "TruncatedSVD",

            "n_components":
                20,

            "random_state":
                42,

            "interaction_value":
                "InteractionStrength",

            "personalized":
                True,

            "cold_start_supported":
                False
        }

    elif (
        selected_model
        ==
        "Hybrid Recommender"
    ):

        model_configuration = {
            "type":
                "HybridRecommender",

            "item_cf_weight":
                0.40,

            "matrix_factorization_weight":
                0.40,

            "popularity_weight":
                0.20,

            "n_components":
                20,

            "random_state":
                42,

            "candidate_multiplier":
                5,

            "personalized":
                True,

            "cold_start_supported":
                True,

            "cold_start_fallback":
                "Popularity"
        }

    else:

        raise ValueError(
            f"Unsupported selected "
            f"recommender: {selected_model}"
        )

    # =================================
    # FINAL METADATA
    # =================================

    metadata = {

        "module":
            "Next Product Recommendation",

        "selected_model":
            selected_model,

        "selection": {

            "k":
                selection[
                    "selection_k"
                ],

            "rule":
                selection[
                    "selection_rule"
                ],

            "metrics":
                selection[
                    "selected_metrics"
                ],

            "evaluation_strategy":
                "Temporal leave-last-invoice-out",

            "ground_truth":
                (
                    "Later customer purchases "
                    "that were available in the "
                    "training product catalog"
                ),

            "repeat_purchases_allowed":
                True
        },

        "model_configuration":
            model_configuration,

        "training_data": {

            "interaction_file":
                (
                    "artifacts/recommendation/"
                    "recommendation_train_interactions.csv"
                ),

            "interaction_value":
                "InteractionStrength"
        },

        "evaluation_artifacts": {

            "complete_comparison":
                comparison_path,

            "final_ranked_comparison":
                ranked_path
        }
    }

    metadata = (
        make_json_serializable(
            metadata
        )
    )

    # =================================
    # SAVE METADATA
    # =================================

    with open(
        selection_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # =================================
    # OUTPUT
    # =================================

    print(
        "\n================================"
    )

    print(
        "FINAL RECOMMENDATION SELECTION"
    )

    print(
        "================================"
    )

    print(
        f"\nSelected Model: "
        f"{selected_model}"
    )

    print(
        f"Selection K: "
        f"{selection['selection_k']}"
    )

    metrics = (
        selection[
            "selected_metrics"
        ]
    )

    print(
        f"\nPrecision@10: "
        f"{metrics['precision_at_k']:.4f}"
    )

    print(
        f"Recall@10: "
        f"{metrics['recall_at_k']:.4f}"
    )

    print(
        f"HitRate@10: "
        f"{metrics['hit_rate_at_k']:.4f}"
    )

    print(
        f"Catalog Coverage: "
        f"{metrics['catalog_coverage']:.4f}"
    )

    print(
        f"Unique Recommended Products: "
        f"{metrics['unique_recommended_products']}"
    )

    print(
        "\n--------------------------------"
    )

    print(
        "K=10 MODEL RANKING"
    )

    print(
        "--------------------------------"
    )

    print(
        "\n"
        + ranked_models.to_string(
            index=False
        )
    )

    print(
        f"\nSelection metadata saved to:"
        f"\n{selection_path}"
    )

    print(
        f"\nRanked comparison saved to:"
        f"\n{ranked_path}"
    )


if __name__ == "__main__":

    main()