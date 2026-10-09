import os

import pandas as pd

from api.recommendation_service import (
    RecommendationService
)


# ============================================================
# BATCH FILE
# ============================================================

def create_batch_file(
    tmp_path
):

    data = pd.DataFrame(
        {
            "Customer ID": [
                101.0,
                102.0
            ],

            "Recommended Products": [
                "Product A|Product B|Product C",
                "Product X|Product Y"
            ],

            "Recommended Stock Codes": [
                "A1|B1|C1",
                "X1|Y1"
            ],

            "Recommendation Scores": [
                "0.9|0.8|0.7",
                "0.95|0.85"
            ],

            "Recommendation Source": [
                "Hybrid",
                "Hybrid"
            ]
        }
    )

    path = os.path.join(
        tmp_path,
        "batch.csv"
    )

    data.to_csv(
        path,
        index=False
    )

    return path


# ============================================================
# NORMALIZE CUSTOMER ID
# ============================================================

def test_normalize_customer_id():

    assert (
        RecommendationService
        .normalize_customer_id(
            101
        )
        ==
        "101"
    )

    assert (
        RecommendationService
        .normalize_customer_id(
            101.0
        )
        ==
        "101"
    )

    assert (
        RecommendationService
        .normalize_customer_id(
            "101.0"
        )
        ==
        "101"
    )


# ============================================================
# PIPE SPLIT
# ============================================================

def test_split_values():

    result = (
        RecommendationService
        .split_values(
            "A|B|C"
        )
    )

    assert result == [
        "A",
        "B",
        "C"
    ]


# ============================================================
# LOAD BATCH
# ============================================================

def test_load_batch_predictions(
    tmp_path
):

    path = create_batch_file(
        tmp_path
    )

    service = RecommendationService(
        model_path="unused.pkl",
        batch_predictions_path=path
    )

    result = (
        service
        .load_batch_predictions()
    )

    assert (
        len(result)
        ==
        2
    )

    assert (
        result.iloc[0][
            "Customer ID"
        ]
        ==
        "101"
    )


# ============================================================
# KNOWN CUSTOMER
# ============================================================

def test_known_customer_recommendation(
    tmp_path
):

    path = create_batch_file(
        tmp_path
    )

    service = RecommendationService(
        model_path="unused.pkl",
        batch_predictions_path=path
    )

    result = (
        service.recommend(
            customer_id="101",
            top_n=2
        )
    )

    assert (
        result[
            "customer_id"
        ]
        ==
        "101"
    )

    assert (
        len(
            result[
                "recommendations"
            ]
        )
        ==
        2
    )

    assert (
        result[
            "recommendations"
        ][0][
            "product"
        ]
        ==
        "Product A"
    )

    assert (
        result[
            "recommendations"
        ][0][
            "stock_code"
        ]
        ==
        "A1"
    )

    assert (
        result[
            "recommendations"
        ][0][
            "score"
        ]
        ==
        0.9
    )

    assert (
        result[
            "cold_start"
        ]
        is False
    )


# ============================================================
# TOP-N LIMIT
# ============================================================

def test_top_n_limit(
    tmp_path
):

    path = create_batch_file(
        tmp_path
    )

    service = RecommendationService(
        model_path="unused.pkl",
        batch_predictions_path=path
    )

    result = (
        service.recommend(
            customer_id="101",
            top_n=1
        )
    )

    assert (
        len(
            result[
                "recommendations"
            ]
        )
        ==
        1
    )


# ============================================================
# INVALID TOP-N
# ============================================================

def test_invalid_top_n(
    tmp_path
):

    path = create_batch_file(
        tmp_path
    )

    service = RecommendationService(
        model_path="unused.pkl",
        batch_predictions_path=path
    )

    try:

        service.recommend(
            customer_id="101",
            top_n=0
        )

        assert False

    except ValueError:

        assert True


# ============================================================
# UNKNOWN CUSTOMER WITHOUT MODEL
# ============================================================

def test_unknown_customer_without_cold_start(
    tmp_path
):

    path = create_batch_file(
        tmp_path
    )

    service = RecommendationService(
        model_path="missing.pkl",
        batch_predictions_path=path
    )

    try:

        service.recommend(
            customer_id="999",
            top_n=5
        )

        assert False

    except LookupError as error:

        assert (
            "No recommendation is available"
            in str(
                error
            )
        )


# ============================================================
# MODEL RESULT NORMALIZATION
# ============================================================

def test_normalize_model_list():

    result = (
        RecommendationService
        ._normalize_model_result(
            [
                {
                    "product":
                        "Product A",

                    "stock_code":
                        "A1",

                    "score":
                        0.90
                },

                {
                    "product":
                        "Product B",

                    "stock_code":
                        "B1",

                    "score":
                        0.80
                }
            ],
            top_n=2
        )
    )

    assert (
        len(result)
        ==
        2
    )

    assert (
        result[0][
            "rank"
        ]
        ==
        1
    )

    assert (
        result[0][
            "product"
        ]
        ==
        "Product A"
    )