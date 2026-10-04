import os

import pandas as pd

from src.explainability.explainability_validation import (
    ExplainabilityValidation
)


# ============================================================
# INSTANCE
# ============================================================

def create_validator():

    return ExplainabilityValidation()


# ============================================================
# FILE CHECK
# ============================================================

def test_check_file(
    tmp_path
):

    validator = create_validator()

    file_path = os.path.join(
        tmp_path,
        "test.txt"
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "hello"
        )

    result = (
        validator.check_file(
            file_path
        )
    )

    assert (
        result[
            "exists"
        ]
        is True
    )

    assert (
        result[
            "size_bytes"
        ]
        >
        0
    )


# ============================================================
# CUSTOMER ID NORMALIZATION
# ============================================================

def test_normalize_customer_id():

    validator = create_validator()

    assert (
        validator.normalize_customer_id(
            17850
        )
        ==
        "17850"
    )

    assert (
        validator.normalize_customer_id(
            17850.0
        )
        ==
        "17850"
    )

    assert (
        validator.normalize_customer_id(
            "17850.0"
        )
        ==
        "17850"
    )


# ============================================================
# IMPORTANCE VALIDATION
# ============================================================

def test_validate_importance_data():

    validator = create_validator()

    data = pd.DataFrame(
        {
            "Feature": [
                "Recency",
                "Frequency"
            ],

            "Mean Absolute SHAP": [
                0.4,
                0.2
            ]
        }
    )

    result = (
        validator
        .validate_importance_data(
            data
        )
    )

    assert (
        result[
            "valid"
        ]
        is True
    )

    assert (
        result[
            "rows"
        ]
        ==
        2
    )


# ============================================================
# LOCAL SHAP VALIDATION
# ============================================================

def test_validate_local_shap_data():

    validator = create_validator()

    data = pd.DataFrame(
        {
            "Customer ID": [
                101,
                101,
                102
            ],

            "Feature": [
                "Recency",
                "Frequency",
                "Recency"
            ],

            "Feature Value": [
                10,
                5,
                100
            ],

            "SHAP Value": [
                -0.2,
                -0.1,
                0.5
            ],

            "Absolute SHAP": [
                0.2,
                0.1,
                0.5
            ]
        }
    )

    result = (
        validator
        .validate_local_shap_data(
            data
        )
    )

    assert (
        result[
            "valid"
        ]
        is True
    )

    assert (
        result[
            "explained_customers"
        ]
        ==
        2
    )


# ============================================================
# TOP FEATURES
# ============================================================

def test_top_features():

    validator = create_validator()

    data = pd.DataFrame(
        {
            "Feature": [
                "Recency",
                "Frequency",
                "Monetary"
            ],

            "Mean Absolute SHAP": [
                0.2,
                0.6,
                0.4
            ]
        }
    )

    result = (
        validator.top_features(
            data,
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
            "Feature"
        ]
        ==
        "Frequency"
    )


# ============================================================
# COVERAGE
# ============================================================

def test_explanation_coverage():

    validator = create_validator()

    shap_data = pd.DataFrame(
        {
            "Customer ID": [
                101,
                101,
                102,
                102
            ]
        }
    )

    customer_data = pd.DataFrame(
        {
            "Customer ID": [
                101,
                102,
                103,
                104
            ]
        }
    )

    result = (
        validator
        .explanation_coverage(
            shap_data,
            customer_data
        )
    )

    assert (
        result[
            "explained_customers"
        ]
        ==
        2
    )

    assert (
        result[
            "total_customers"
        ]
        ==
        4
    )

    assert (
        result[
            "coverage_percentage"
        ]
        ==
        50.0
    )


# ============================================================
# COMMON CUSTOMERS
# ============================================================

def test_common_explained_customers():

    validator = create_validator()

    churn = pd.DataFrame(
        {
            "Customer ID": [
                101,
                102,
                103
            ]
        }
    )

    clv = pd.DataFrame(
        {
            "Customer ID": [
                102,
                103,
                104
            ]
        }
    )

    result = (
        validator
        .common_explained_customers(
            churn,
            clv
        )
    )

    assert (
        result[
            "count"
        ]
        ==
        2
    )

    assert (
        result[
            "customers"
        ]
        ==
        [
            "102",
            "103"
        ]
    )