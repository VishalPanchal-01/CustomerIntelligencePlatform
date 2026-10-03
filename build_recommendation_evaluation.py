import json
import os

import pandas as pd

from src.evaluation.recommendation_temporal_split import (
    RecommendationTemporalSplitter
)

from src.evaluation.recommendation_ground_truth import (
    RecommendationGroundTruthBuilder
)

from src.feature_engineering.recommendation_interactions import (
    RecommendationInteractionBuilder
)


def main():

    # =================================
    # PATHS
    # =================================

    input_path = (
        "artifacts/recommendation/"
        "clean_purchase_transactions.csv"
    )

    output_directory = (
        "artifacts/recommendation"
    )

    train_transaction_path = os.path.join(
        output_directory,
        "recommendation_train_transactions.csv"
    )

    train_interaction_path = os.path.join(
        output_directory,
        "recommendation_train_interactions.csv"
    )

    test_transaction_path = os.path.join(
        output_directory,
        "recommendation_test_transactions.csv"
    )

    ground_truth_path = os.path.join(
        output_directory,
        "recommendation_test_ground_truth.csv"
    )

    evaluable_ground_truth_path = os.path.join(
        output_directory,
        "recommendation_evaluable_ground_truth.csv"
    )

    summary_path = os.path.join(
        output_directory,
        "recommendation_split_summary.json"
    )

    # =================================
    # LOAD CLEAN TRANSACTIONS
    # =================================

    df = pd.read_csv(
        input_path
    )

    df[
        "InvoiceDate"
    ] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    # =================================
    # TEMPORAL SPLIT
    # =================================

    splitter = (
        RecommendationTemporalSplitter(
            holdout_invoices=1,
            minimum_invoices=2
        )
    )

    (
        train_transactions,
        test_transactions,
        eligible_customers
    ) = splitter.split(
        df
    )

    # =================================
    # BUILD TRAIN INTERACTIONS ONLY
    # =================================

    interaction_builder = (
        RecommendationInteractionBuilder()
    )

    train_interactions = (
        interaction_builder.build_interactions(
            train_transactions
        )
    )

    # =================================
    # BUILD TEST GROUND TRUTH
    # =================================

    ground_truth_builder = (
        RecommendationGroundTruthBuilder()
    )

    (
        ground_truth,
        evaluable_ground_truth
    ) = (
        ground_truth_builder
        .build_ground_truth(
            test_transactions=
                test_transactions,

            train_transactions=
                train_transactions
        )
    )

    # =================================
    # EVALUATION CUSTOMER COUNTS
    # =================================

    heldout_customers = (
        ground_truth[
            "CustomerID"
        ]
        .nunique()
    )

    evaluable_customers = (
        evaluable_ground_truth[
            "CustomerID"
        ]
        .nunique()
    )

    all_test_items = len(
        ground_truth
    )

    evaluable_test_items = len(
        evaluable_ground_truth
    )

    cold_start_test_items = (
        all_test_items
        -
        evaluable_test_items
    )

    if all_test_items > 0:

        evaluable_item_percentage = (
            evaluable_test_items
            /
            all_test_items
            *
            100
        )

    else:

        evaluable_item_percentage = 0.0

    # =================================
    # TRAINING DATA STATISTICS
    # =================================

    train_customers = (
        train_transactions[
            "CustomerID"
        ]
        .nunique()
    )

    train_products = (
        train_transactions[
            "StockCode"
        ]
        .nunique()
    )

    train_interaction_count = (
        len(
            train_interactions
        )
    )

    possible_pairs = (
        train_customers
        *
        train_products
    )

    if possible_pairs > 0:

        train_density = (
            train_interaction_count
            /
            possible_pairs
            *
            100
        )

    else:

        train_density = 0.0

    # =================================
    # SUMMARY
    # =================================

    summary = {

        "split_strategy":
            "Leave-last-invoice-out",

        "holdout_invoices_per_eligible_customer":
            1,

        "minimum_invoices_for_evaluation":
            2,

        "training": {

            "transaction_rows":
                int(
                    len(
                        train_transactions
                    )
                ),

            "customers":
                int(
                    train_customers
                ),

            "products":
                int(
                    train_products
                ),

            "customer_product_interactions":
                int(
                    train_interaction_count
                ),

            "interaction_matrix_density_percentage":
                float(
                    train_density
                )
        },

        "evaluation": {

            "eligible_customers":
                int(
                    len(
                        eligible_customers
                    )
                ),

            "customers_with_heldout_products":
                int(
                    heldout_customers
                ),

            "customers_with_evaluable_ground_truth":
                int(
                    evaluable_customers
                ),

            "heldout_customer_product_pairs":
                int(
                    all_test_items
                ),

            "evaluable_customer_product_pairs":
                int(
                    evaluable_test_items
                ),

            "cold_start_product_pairs":
                int(
                    cold_start_test_items
                ),

            "evaluable_item_percentage":
                float(
                    evaluable_item_percentage
                )
        },

        "leakage_control": {

            "split_level":
                "Invoice",

            "future_invoices_removed_from_training":
                True,

            "training_interactions_built_after_split":
                True
        }
    }

    # =================================
    # SAVE FILES
    # =================================

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    train_transactions.to_csv(
        train_transaction_path,
        index=False
    )

    train_interactions.to_csv(
        train_interaction_path,
        index=False
    )

    test_transactions.to_csv(
        test_transaction_path,
        index=False
    )

    ground_truth.to_csv(
        ground_truth_path,
        index=False
    )

    evaluable_ground_truth.to_csv(
        evaluable_ground_truth_path,
        index=False
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    # =================================
    # CONSOLE OUTPUT
    # =================================

    print(
        "\n================================"
    )

    print(
        "RECOMMENDATION TEMPORAL SPLIT"
    )

    print(
        "================================"
    )

    print(
        f"\nTraining rows: "
        f"{len(train_transactions)}"
    )

    print(
        f"Training interactions: "
        f"{len(train_interactions)}"
    )

    print(
        f"Evaluation customers: "
        f"{len(eligible_customers)}"
    )

    print(
        f"Held-out product pairs: "
        f"{all_test_items}"
    )

    print(
        f"Evaluable product pairs: "
        f"{evaluable_test_items}"
    )

    print(
        f"Cold-start test products: "
        f"{cold_start_test_items}"
    )

    print(
        f"Evaluable item percentage: "
        f"{evaluable_item_percentage:.2f}%"
    )

    print(
        f"\nTraining interactions saved to:"
        f"\n{train_interaction_path}"
    )

    print(
        f"\nEvaluation ground truth saved to:"
        f"\n{evaluable_ground_truth_path}"
    )

    print(
        f"\nSplit summary saved to:"
        f"\n{summary_path}"
    )


if __name__ == "__main__":

    main()