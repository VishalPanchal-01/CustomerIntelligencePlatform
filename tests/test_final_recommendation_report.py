import json
import os

import pandas as pd

from src.evaluation.final_recommendation_report import (
    FinalRecommendationReport
)


def test_final_recommendation_report(
    tmp_path
):

    selection_metadata = {

        "selected_model":
            "Hybrid Recommender",

        "selection": {

            "k":
                10,

            "rule": {

                "primary_metric":
                    "Recall@10",

                "tie_breaker_1":
                    "HitRate@10",

                "tie_breaker_2":
                    "Precision@10",

                "tie_breaker_3":
                    "CatalogCoverage"
            },

            "metrics": {

                "precision_at_k":
                    0.08,

                "recall_at_k":
                    0.40,

                "hit_rate_at_k":
                    0.50,

                "catalog_coverage":
                    0.25,

                "unique_recommended_products":
                    200
            },

            "repeat_purchases_allowed":
                True
        },

        "model_configuration": {

            "type":
                "HybridRecommender",

            "item_cf_weight":
                0.4,

            "matrix_factorization_weight":
                0.4,

            "popularity_weight":
                0.2,

            "n_components":
                20,

            "random_state":
                42,

            "candidate_multiplier":
                5,

            "cold_start_supported":
                True
        }
    }

    production_metadata = {

        "selected_model":
            "Hybrid Recommender",

        "model_path":
            (
                "models/recommendation/"
                "recommender.pkl"
            ),

        "training_source":
            (
                "artifacts/recommendation/"
                "customer_product_interactions.csv"
            ),

        "training_scope":
            (
                "Complete valid customer-product "
                "interaction history"
            ),

        "evaluation_scope":
            (
                "Metrics were obtained from "
                "temporal evaluation."
            ),

        "dataset": {

            "interaction_count":
                50000,

            "customer_count":
                4000,

            "product_count":
                3000
        },

        "prediction": {

            "default_top_k":
                10,

            "repeat_purchase_mode_supported":
                True,

            "discovery_mode_supported":
                True,

            "cold_start_support":
                True
        }
    }

    model_comparison = pd.DataFrame(
        {
            "Model": [
                "Popularity Baseline",
                "Item-Based Collaborative Filtering",
                "Matrix Factorization",
                "Hybrid Recommender"
            ],

            "K": [
                10,
                10,
                10,
                10
            ],

            "PrecisionAtK": [
                0.03,
                0.05,
                0.06,
                0.08
            ],

            "RecallAtK": [
                0.15,
                0.30,
                0.35,
                0.40
            ],

            "HitRateAtK": [
                0.20,
                0.35,
                0.40,
                0.50
            ],

            "CatalogCoverage": [
                0.01,
                0.20,
                0.30,
                0.25
            ],

            "UniqueRecommendedProducts": [
                10,
                150,
                220,
                200
            ]
        }
    )

    split_summary = {

        "split_strategy":
            "Leave-last-invoice-out",

        "holdout_invoices_per_eligible_customer":
            1,

        "minimum_invoices_for_evaluation":
            2,

        "training": {

            "transaction_rows":
                100000,

            "customers":
                4500,

            "products":
                3000
        },

        "evaluation": {

            "eligible_customers":
                2500,

            "customers_with_evaluable_ground_truth":
                2400
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

    output_path = os.path.join(
        tmp_path,
        "recommendation_model_report.json"
    )

    generator = (
        FinalRecommendationReport()
    )

    report = (
        generator.generate_report(
            selection_metadata=
                selection_metadata,

            production_metadata=
                production_metadata,

            model_comparison=
                model_comparison,

            split_summary=
                split_summary,

            output_path=
                output_path
        )
    )

    assert report is not None

    assert os.path.exists(
        output_path
    )

    assert (
        report[
            "module"
        ][
            "problem_type"
        ]
        ==
        "Implicit-feedback Top-N recommendation"
    )

    assert (
        report[
            "selected_strategy"
        ][
            "model"
        ]
        ==
        "Hybrid Recommender"
    )

    assert (
        report[
            "selected_strategy"
        ][
            "recall_at_k"
        ]
        ==
        0.40
    )

    assert (
        report[
            "production_prediction"
        ][
            "default_top_k"
        ]
        ==
        10
    )

    assert (
        report[
            "production_prediction"
        ][
            "discovery_mode"
        ][
            "supported"
        ]
        is True
    )

    assert (
        len(
            report[
                "candidate_models"
            ]
        )
        ==
        4
    )

    with open(
        output_path,
        "r",
        encoding="utf-8"
    ) as file:

        saved_report = (
            json.load(
                file
            )
        )

    assert (
        saved_report[
            "evaluation_design"
        ][
            "strategy"
        ]
        ==
        "Leave-last-invoice-out"
    )