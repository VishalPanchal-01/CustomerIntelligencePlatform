import json
import os

import pandas as pd

from src.training.final_recommendation_trainer import (
    FinalRecommendationTrainer
)

from src.utils.recommendation_persistence import (
    RecommendationPersistence
)


def main():

    # =================================
    # PATHS
    # =================================

    selection_path = (
        "artifacts/recommendation/"
        "final/"
        "recommendation_final_selection.json"
    )

    interaction_path = (
        "artifacts/recommendation/"
        "customer_product_interactions.csv"
    )

    model_path = (
        "models/recommendation/"
        "recommender.pkl"
    )

    metadata_path = (
        "models/recommendation/"
        "recommender_metadata.json"
    )

    # =================================
    # VALIDATE INPUT FILES
    # =================================

    required_files = [
        selection_path,
        interaction_path
    ]

    for file_path in required_files:

        if not os.path.exists(
            file_path
        ):

            raise FileNotFoundError(
                f"Required recommendation file "
                f"not found: {file_path}"
            )

    # =================================
    # LOAD SELECTION METADATA
    # =================================

    with open(
        selection_path,
        "r",
        encoding="utf-8"
    ) as file:

        selection_metadata = (
            json.load(
                file
            )
        )

    selected_model = (
        selection_metadata[
            "selected_model"
        ]
    )

    configuration = (
        selection_metadata[
            "model_configuration"
        ]
    )

    # =================================
    # LOAD FULL INTERACTION HISTORY
    # =================================

    interactions = pd.read_csv(
        interaction_path
    )

    print(
        "\n================================"
    )

    print(
        "FINAL RECOMMENDER TRAINING"
    )

    print(
        "================================"
    )

    print(
        f"\nSelected model: "
        f"{selected_model}"
    )

    print(
        f"Production interactions: "
        f"{len(interactions)}"
    )

    print(
        f"Customers: "
        f"{interactions['CustomerID'].nunique()}"
    )

    print(
        f"Products: "
        f"{interactions['StockCode'].nunique()}"
    )

    # =================================
    # TRAIN SELECTED RECOMMENDER
    # =================================

    trainer = (
        FinalRecommendationTrainer()
    )

    recommender = (
        trainer.train(
            interactions=
                interactions,

            selected_model=
                selected_model,

            configuration=
                configuration
        )
    )

    # =================================
    # SAVE MODEL
    # =================================

    persistence = (
        RecommendationPersistence()
    )

    persistence.save_model(
        recommender,
        model_path
    )

    # =================================
    # RELOAD VERIFICATION
    # =================================

    loaded_model = (
        persistence.load_model(
            model_path
        )
    )

    # ---------------------------------
    # Pick one known customer
    # ---------------------------------

    sample_customer = (
        interactions[
            "CustomerID"
        ]
        .iloc[0]
    )

    recommendations = (
        loaded_model.recommend(
            customer_id=
                sample_customer,

            top_k=
                5,

            exclude_already_purchased=
                False
        )
    )

    # =================================
    # PRODUCTION METADATA
    # =================================

    metadata = {

        "module":
            "Next Product Recommendation",

        "selected_model":
            selected_model,

        "model_path":
            model_path,

        "training_source":
            interaction_path,

        "training_scope":
            (
                "Complete valid customer-product "
                "interaction history available "
                "after model selection"
            ),

        "evaluation_scope":
            (
                "Model selection metrics were obtained "
                "from temporal leave-last-invoice-out "
                "evaluation before full-history retraining."
            ),

        "configuration":
            configuration,

        "dataset": {

            "interaction_count":
                int(
                    len(
                        interactions
                    )
                ),

            "customer_count":
                int(
                    interactions[
                        "CustomerID"
                    ]
                    .nunique()
                ),

            "product_count":
                int(
                    interactions[
                        "StockCode"
                    ]
                    .nunique()
                )
        },

        "prediction": {

            "default_top_k":
                10,

            "repeat_purchase_mode_supported":
                True,

            "discovery_mode_supported":
                True,

            "cold_start_support":
                bool(
                    configuration.get(
                        "cold_start_supported",
                        False
                    )
                )
        },

        "persistence_verification": {

            "reload_successful":
                True,

            "sample_customer":
                str(
                    sample_customer
                ),

            "sample_recommendation_count":
                int(
                    len(
                        recommendations
                    )
                )
        }
    }

    # =================================
    # SAVE METADATA
    # =================================

    os.makedirs(
        os.path.dirname(
            metadata_path
        ),
        exist_ok=True
    )

    with open(
        metadata_path,
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
        f"\nModel saved to:"
        f"\n{model_path}"
    )

    print(
        f"\nMetadata saved to:"
        f"\n{metadata_path}"
    )

    print(
        "\n--------------------------------"
    )

    print(
        "RELOAD VERIFICATION"
    )

    print(
        "--------------------------------"
    )

    print(
        f"\nSample Customer: "
        f"{sample_customer}"
    )

    print(
        f"Recommendations returned: "
        f"{len(recommendations)}"
    )

    if not recommendations.empty:

        print(
            "\n"
            + recommendations.head(
                5
            )
            .to_string(
                index=False
            )
        )


if __name__ == "__main__":

    main()