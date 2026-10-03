import json
import os
import sys

import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class FinalRecommendationReport:

    def generate_report(
        self,
        selection_metadata: dict,
        production_metadata: dict,
        model_comparison: pd.DataFrame,
        split_summary: dict,
        output_path: str
    ) -> dict:

        try:

            logger.info(
                "Starting final recommendation "
                "report generation."
            )

            # =================================
            # VALIDATE INPUT
            # =================================

            if not selection_metadata:

                raise ValueError(
                    "Recommendation selection "
                    "metadata is empty."
                )

            if not production_metadata:

                raise ValueError(
                    "Recommendation production "
                    "metadata is empty."
                )

            if model_comparison.empty:

                raise ValueError(
                    "Recommendation model "
                    "comparison is empty."
                )

            if not split_summary:

                raise ValueError(
                    "Recommendation split "
                    "summary is empty."
                )

            required_comparison_columns = [
                "Model",
                "K",
                "PrecisionAtK",
                "RecallAtK",
                "HitRateAtK",
                "CatalogCoverage"
            ]

            missing_columns = [
                column
                for column in required_comparison_columns
                if column not in model_comparison.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing recommendation comparison "
                    f"columns: {missing_columns}"
                )

            # =================================
            # ALL MODEL RESULTS
            # =================================

            comparison_records = []

            sorted_comparison = (
                model_comparison
                .sort_values(
                    by=[
                        "K",
                        "Model"
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            for _, row in (
                sorted_comparison
                .iterrows()
            ):

                record = {

                    "model":
                        str(
                            row[
                                "Model"
                            ]
                        ),

                    "k":
                        int(
                            row[
                                "K"
                            ]
                        ),

                    "precision_at_k":
                        float(
                            row[
                                "PrecisionAtK"
                            ]
                        ),

                    "recall_at_k":
                        float(
                            row[
                                "RecallAtK"
                            ]
                        ),

                    "hit_rate_at_k":
                        float(
                            row[
                                "HitRateAtK"
                            ]
                        ),

                    "catalog_coverage":
                        float(
                            row[
                                "CatalogCoverage"
                            ]
                        )
                }

                if (
                    "UniqueRecommendedProducts"
                    in row.index
                ):

                    record[
                        "unique_recommended_products"
                    ] = int(
                        row[
                            "UniqueRecommendedProducts"
                        ]
                    )

                comparison_records.append(
                    record
                )

            # =================================
            # SELECTED MODEL DETAILS
            # =================================

            selected_model = (
                selection_metadata[
                    "selected_model"
                ]
            )

            selection_info = (
                selection_metadata[
                    "selection"
                ]
            )

            selected_metrics = (
                selection_info[
                    "metrics"
                ]
            )

            model_configuration = (
                selection_metadata[
                    "model_configuration"
                ]
            )

            # =================================
            # BUILD FINAL REPORT
            # =================================

            report = {

                "module": {

                    "name":
                        "Next Product Recommendation",

                    "problem_type":
                        (
                            "Implicit-feedback "
                            "Top-N recommendation"
                        ),

                    "business_goal":
                        (
                            "Recommend products that a "
                            "customer is likely to purchase "
                            "next, while also supporting "
                            "product-discovery mode."
                        )
                },

                "interaction_engineering": {

                    "source":
                        (
                            "Clean positive purchase "
                            "transactions"
                        ),

                    "interaction_unit":
                        "CustomerID x StockCode",

                    "aggregated_features": [
                        "PurchaseCount",
                        "TotalQuantity",
                        "TotalSpend",
                        "LastPurchaseDate",
                        "InteractionStrength"
                    ],

                    "interaction_strength_formula":
                        (
                            "log1p(PurchaseCount) + "
                            "log1p(TotalQuantity)"
                        ),

                    "explicit_ratings_used":
                        False,

                    "feedback_type":
                        "Implicit",

                    "cancellations_as_positive_signal":
                        False,

                    "negative_quantity_as_positive_signal":
                        False
                },

                "evaluation_design": {

                    "strategy":
                        (
                            split_summary.get(
                                "split_strategy",
                                "Leave-last-invoice-out"
                            )
                        ),

                    "split_level":
                        (
                            split_summary
                            .get(
                                "leakage_control",
                                {}
                            )
                            .get(
                                "split_level",
                                "Invoice"
                            )
                        ),

                    "holdout_invoices_per_customer":
                        (
                            split_summary.get(
                                "holdout_invoices_per_eligible_customer",
                                1
                            )
                        ),

                    "minimum_invoices_for_evaluation":
                        (
                            split_summary.get(
                                "minimum_invoices_for_evaluation",
                                2
                            )
                        ),

                    "future_interactions_removed_before_training":
                        (
                            split_summary
                            .get(
                                "leakage_control",
                                {}
                            )
                            .get(
                                "future_invoices_removed_from_training",
                                True
                            )
                        ),

                    "training_interactions_built_after_split":
                        (
                            split_summary
                            .get(
                                "leakage_control",
                                {}
                            )
                            .get(
                                "training_interactions_built_after_split",
                                True
                            )
                        ),

                    "warm_item_evaluation":
                        True,

                    "cold_start_products_separated":
                        True,

                    "repeat_purchases_allowed_during_evaluation":
                        bool(
                            selection_info[
                                "repeat_purchases_allowed"
                            ]
                        )
                },

                "evaluation_population": {

                    "training":
                        split_summary.get(
                            "training",
                            {}
                        ),

                    "evaluation":
                        split_summary.get(
                            "evaluation",
                            {}
                        )
                },

                "candidate_models": [

                    {
                        "name":
                            "Popularity Baseline",

                        "approach":
                            (
                                "Global product popularity "
                                "using customer adoption, "
                                "purchase frequency and quantity"
                            )
                    },

                    {
                        "name":
                            (
                                "Item-Based Collaborative "
                                "Filtering"
                            ),

                        "approach":
                            (
                                "Customer-item sparse matrix "
                                "with item-item cosine similarity"
                            )
                    },

                    {
                        "name":
                            "Matrix Factorization",

                        "approach":
                            (
                                "TruncatedSVD over the "
                                "implicit customer-product "
                                "interaction matrix"
                            )
                    },

                    {
                        "name":
                            "Hybrid Recommender",

                        "approach":
                            (
                                "Weighted fusion of Item-CF, "
                                "Matrix Factorization and "
                                "Popularity scores"
                            )
                    }
                ],

                "evaluation_metrics": {

                    "metrics_used": [
                        "Precision@K",
                        "Recall@K",
                        "HitRate@K",
                        "CatalogCoverage"
                    ],

                    "evaluated_k_values": [
                        5,
                        10,
                        20
                    ],

                    "primary_selection_metric":
                        selection_info[
                            "rule"
                        ][
                            "primary_metric"
                        ],

                    "tie_breaker_1":
                        selection_info[
                            "rule"
                        ][
                            "tie_breaker_1"
                        ],

                    "tie_breaker_2":
                        selection_info[
                            "rule"
                        ][
                            "tie_breaker_2"
                        ],

                    "tie_breaker_3":
                        selection_info[
                            "rule"
                        ][
                            "tie_breaker_3"
                        ]
                },

                "model_comparison":
                    comparison_records,

                "selected_strategy": {

                    "model":
                        selected_model,

                    "selection_k":
                        int(
                            selection_info[
                                "k"
                            ]
                        ),

                    "precision_at_k":
                        float(
                            selected_metrics[
                                "precision_at_k"
                            ]
                        ),

                    "recall_at_k":
                        float(
                            selected_metrics[
                                "recall_at_k"
                            ]
                        ),

                    "hit_rate_at_k":
                        float(
                            selected_metrics[
                                "hit_rate_at_k"
                            ]
                        ),

                    "catalog_coverage":
                        float(
                            selected_metrics[
                                "catalog_coverage"
                            ]
                        ),

                    "unique_recommended_products":
                        int(
                            selected_metrics[
                                "unique_recommended_products"
                            ]
                        ),

                    "configuration":
                        model_configuration
                },

                "production_training": {

                    "retrained_after_selection":
                        True,

                    "training_scope":
                        production_metadata[
                            "training_scope"
                        ],

                    "training_source":
                        production_metadata[
                            "training_source"
                        ],

                    "evaluation_scope_note":
                        production_metadata[
                            "evaluation_scope"
                        ],

                    "interaction_count":
                        int(
                            production_metadata[
                                "dataset"
                            ][
                                "interaction_count"
                            ]
                        ),

                    "customer_count":
                        int(
                            production_metadata[
                                "dataset"
                            ][
                                "customer_count"
                            ]
                        ),

                    "product_count":
                        int(
                            production_metadata[
                                "dataset"
                            ][
                                "product_count"
                            ]
                        )
                },

                "production_prediction": {

                    "model_path":
                        production_metadata[
                            "model_path"
                        ],

                    "model_loaded_once":
                        True,

                    "default_top_k":
                        int(
                            production_metadata[
                                "prediction"
                            ][
                                "default_top_k"
                            ]
                        ),

                    "next_purchase_mode":
                        {
                            "supported":
                                True,

                            "already_purchased_products_excluded":
                                False
                        },

                    "discovery_mode":
                        {
                            "supported":
                                True,

                            "already_purchased_products_excluded":
                                True
                        },

                    "batch_prediction_supported":
                        True,

                    "standard_output_fields": [
                        "StockCode",
                        "Description",
                        "Score",
                        "Rank"
                    ]
                },

                "cold_start": {

                    "user_cold_start":
                        (
                            "Popularity fallback is used when "
                            "available in the persisted "
                            "production strategy."
                        ),

                    "item_cold_start":
                        (
                            "Products not seen during model "
                            "training cannot receive learned "
                            "collaborative or latent-factor "
                            "scores."
                        ),

                    "evaluation_treatment":
                        (
                            "Held-out products unavailable in "
                            "the training catalog are tracked "
                            "separately from warm-item metrics."
                        )
                },

                "interpretation": {

                    "recommendation_score":
                        (
                            "The recommendation score is a "
                            "ranking score. It is not a "
                            "purchase probability."
                        ),

                    "precision":
                        (
                            "Fraction of Top-K recommendation "
                            "slots matching held-out products."
                        ),

                    "recall":
                        (
                            "Fraction of the customer's "
                            "held-out products recovered "
                            "inside Top-K."
                        ),

                    "hit_rate":
                        (
                            "Fraction of evaluated customers "
                            "for whom at least one held-out "
                            "product appears in Top-K."
                        ),

                    "catalog_coverage":
                        (
                            "Fraction of the training product "
                            "catalog appearing across generated "
                            "recommendations."
                        )
                },

                "limitations": [

                    (
                        "The same temporal benchmark was used "
                        "for comparing recommendation "
                        "configurations; there is not yet a "
                        "separate later untouched temporal "
                        "test period."
                    ),

                    (
                        "Hybrid weights and matrix "
                        "factorization dimensions are "
                        "evaluated configurations rather "
                        "than globally optimal parameters."
                    ),

                    (
                        "Recommendation scores are ranking "
                        "signals and are not calibrated "
                        "purchase probabilities."
                    ),

                    (
                        "The saved recommender represents "
                        "customer behavior available at "
                        "training time and requires retraining "
                        "or an incremental strategy to learn "
                        "new interactions."
                    ),

                    (
                        "New products without historical "
                        "interactions remain an item "
                        "cold-start challenge."
                    )
                ],

                "artifacts": {

                    "full_interactions":
                        (
                            "artifacts/recommendation/"
                            "customer_product_interactions.csv"
                        ),

                    "training_interactions":
                        (
                            "artifacts/recommendation/"
                            "recommendation_train_interactions.csv"
                        ),

                    "ground_truth":
                        (
                            "artifacts/recommendation/"
                            "recommendation_test_ground_truth.csv"
                        ),

                    "evaluable_ground_truth":
                        (
                            "artifacts/recommendation/"
                            "recommendation_evaluable_ground_truth.csv"
                        ),

                    "model_comparison":
                        (
                            "artifacts/recommendation/"
                            "recommendation_model_comparison.csv"
                        ),

                    "selection_metadata":
                        (
                            "artifacts/recommendation/final/"
                            "recommendation_final_selection.json"
                        ),

                    "production_model":
                        production_metadata[
                            "model_path"
                        ],

                    "production_metadata":
                        (
                            "models/recommendation/"
                            "recommender_metadata.json"
                        )
                }
            }

            # =================================
            # OUTPUT DIRECTORY
            # =================================

            output_directory = (
                os.path.dirname(
                    output_path
                )
            )

            if output_directory:

                os.makedirs(
                    output_directory,
                    exist_ok=True
                )

            # =================================
            # SAVE
            # =================================

            with open(
                output_path,
                "w",
                encoding="utf-8"
            ) as file:

                json.dump(
                    report,
                    file,
                    indent=4
                )

            logger.info(
                f"Final recommendation report saved: "
                f"{output_path}"
            )

            return report

        except Exception as e:

            logger.error(
                "Final recommendation report "
                "generation failed."
            )

            raise CustomException(
                e,
                sys
            )