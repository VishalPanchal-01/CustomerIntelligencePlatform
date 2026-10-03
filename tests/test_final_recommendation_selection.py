import pandas as pd

from src.evaluation.final_recommendation_selection import (
    FinalRecommendationSelector
)


def create_comparison_data():

    return pd.DataFrame(
        {
            "Model": [
                "Popularity Baseline",
                "Item-Based Collaborative Filtering",
                "Matrix Factorization",
                "Hybrid Recommender",

                "Popularity Baseline",
                "Item-Based Collaborative Filtering",
                "Matrix Factorization",
                "Hybrid Recommender"
            ],

            "K": [
                5,
                5,
                5,
                5,

                10,
                10,
                10,
                10
            ],

            "PrecisionAtK": [
                0.02,
                0.04,
                0.05,
                0.06,

                0.03,
                0.05,
                0.06,
                0.07
            ],

            "RecallAtK": [
                0.10,
                0.20,
                0.25,
                0.30,

                0.15,
                0.30,
                0.35,
                0.40
            ],

            "HitRateAtK": [
                0.12,
                0.24,
                0.30,
                0.36,

                0.18,
                0.35,
                0.40,
                0.45
            ],

            "CatalogCoverage": [
                0.01,
                0.15,
                0.20,
                0.18,

                0.02,
                0.20,
                0.30,
                0.25
            ],

            "UniqueRecommendedProducts": [
                5,
                100,
                150,
                130,

                10,
                150,
                220,
                180
            ]
        }
    )


def test_final_recommendation_selection():

    comparison = (
        create_comparison_data()
    )

    selector = (
        FinalRecommendationSelector(
            selection_k=10
        )
    )

    result = (
        selector.select_model(
            comparison
        )
    )

    assert result is not None

    assert (
        result[
            "selected_model"
        ]
        ==
        "Hybrid Recommender"
    )

    assert (
        result[
            "selection_k"
        ]
        ==
        10
    )

    assert (
        result[
            "selected_metrics"
        ][
            "recall_at_k"
        ]
        ==
        0.40
    )

    assert (
        result[
            "ranked_models"
        ][
            0
        ][
            "Model"
        ]
        ==
        "Hybrid Recommender"
    )


def test_recommendation_selection_tie_breaker():

    comparison = pd.DataFrame(
        {
            "Model": [
                "Model A",
                "Model B"
            ],

            "K": [
                10,
                10
            ],

            "PrecisionAtK": [
                0.05,
                0.06
            ],

            "RecallAtK": [
                0.30,
                0.30
            ],

            "HitRateAtK": [
                0.35,
                0.40
            ],

            "CatalogCoverage": [
                0.30,
                0.20
            ],

            "UniqueRecommendedProducts": [
                200,
                150
            ]
        }
    )

    selector = (
        FinalRecommendationSelector(
            selection_k=10
        )
    )

    result = (
        selector.select_model(
            comparison
        )
    )

    # Same Recall.
    # Model B has higher HitRate.
    assert (
        result[
            "selected_model"
        ]
        ==
        "Model B"
    )


def test_recommendation_selection_coverage_tie_breaker():

    comparison = pd.DataFrame(
        {
            "Model": [
                "Model A",
                "Model B"
            ],

            "K": [
                10,
                10
            ],

            "PrecisionAtK": [
                0.05,
                0.05
            ],

            "RecallAtK": [
                0.30,
                0.30
            ],

            "HitRateAtK": [
                0.40,
                0.40
            ],

            "CatalogCoverage": [
                0.25,
                0.40
            ],

            "UniqueRecommendedProducts": [
                100,
                160
            ]
        }
    )

    selector = (
        FinalRecommendationSelector(
            selection_k=10
        )
    )

    result = (
        selector.select_model(
            comparison
        )
    )

    # Recall, HitRate and Precision are tied.
    # Model B wins because of coverage.
    assert (
        result[
            "selected_model"
        ]
        ==
        "Model B"
    )