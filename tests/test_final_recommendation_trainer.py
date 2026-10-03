import pandas as pd

from src.recommendation.popularity_recommender import (
    PopularityRecommender
)

from src.recommendation.item_based_cf import (
    ItemBasedCollaborativeRecommender
)

from src.recommendation.matrix_factorization import (
    MatrixFactorizationRecommender
)

from src.recommendation.hybrid_recommender import (
    HybridRecommender
)

from src.training.final_recommendation_trainer import (
    FinalRecommendationTrainer
)


def create_training_data():

    return pd.DataFrame(
        {
            "CustomerID": [
                101,
                101,
                102,
                102,
                103,
                103,
                104,
                104
            ],

            "StockCode": [
                "P1",
                "P2",
                "P1",
                "P3",
                "P2",
                "P3",
                "P3",
                "P4"
            ],

            "Description": [
                "Product 1",
                "Product 2",
                "Product 1",
                "Product 3",
                "Product 2",
                "Product 3",
                "Product 3",
                "Product 4"
            ],

            "PurchaseCount": [
                3,
                2,
                3,
                2,
                2,
                3,
                2,
                1
            ],

            "TotalQuantity": [
                10,
                5,
                9,
                6,
                5,
                8,
                6,
                3
            ],

            "TotalSpend": [
                100,
                50,
                90,
                60,
                50,
                80,
                60,
                30
            ],

            "InteractionStrength": [
                4.0,
                3.0,
                3.8,
                3.1,
                3.0,
                3.7,
                3.2,
                2.0
            ]
        }
    )


def test_build_popularity_model():

    trainer = (
        FinalRecommendationTrainer()
    )

    model = (
        trainer.build_model(
            selected_model=
                "Popularity Baseline",

            configuration={}
        )
    )

    assert isinstance(
        model,
        PopularityRecommender
    )


def test_build_item_cf_model():

    trainer = (
        FinalRecommendationTrainer()
    )

    model = (
        trainer.build_model(
            selected_model=
                "Item-Based Collaborative Filtering",

            configuration={}
        )
    )

    assert isinstance(
        model,
        ItemBasedCollaborativeRecommender
    )


def test_build_matrix_factorization_model():

    trainer = (
        FinalRecommendationTrainer()
    )

    model = (
        trainer.build_model(
            selected_model=
                "Matrix Factorization",

            configuration={
                "n_components":
                    2,

                "random_state":
                    42
            }
        )
    )

    assert isinstance(
        model,
        MatrixFactorizationRecommender
    )

    assert (
        model.n_components
        ==
        2
    )


def test_build_hybrid_model():

    trainer = (
        FinalRecommendationTrainer()
    )

    model = (
        trainer.build_model(
            selected_model=
                "Hybrid Recommender",

            configuration={
                "item_cf_weight":
                    0.4,

                "matrix_factorization_weight":
                    0.4,

                "popularity_weight":
                    0.2,

                "n_components":
                    2,

                "random_state":
                    42,

                "candidate_multiplier":
                    5
            }
        )
    )

    assert isinstance(
        model,
        HybridRecommender
    )


def test_train_final_recommender():

    interactions = (
        create_training_data()
    )

    trainer = (
        FinalRecommendationTrainer()
    )

    model = (
        trainer.train(
            interactions=
                interactions,

            selected_model=
                "Hybrid Recommender",

            configuration={
                "item_cf_weight":
                    0.4,

                "matrix_factorization_weight":
                    0.4,

                "popularity_weight":
                    0.2,

                "n_components":
                    2,

                "random_state":
                    42,

                "candidate_multiplier":
                    5
            }
        )
    )

    recommendations = (
        model.recommend(
            customer_id=101,
            top_k=3
        )
    )

    assert (
        recommendations
        is not None
    )

    assert (
        len(
            recommendations
        )
        > 0
    )