import os

import pandas as pd

from api.customer_service import (
    CustomerIntelligenceService
)


# ============================================================
# TEST DATA
# ============================================================

def create_test_csv(
    tmp_path
):

    data = pd.DataFrame(
        {
            "Customer ID": [
                101.0,
                102.0,
                103.0
            ],

            "Customer Segment": [
                "Segment 0",
                "Segment 1",
                "Segment 2"
            ],

            "Churn Probability": [
                0.8,
                0.4,
                0.1
            ],

            "Churn Risk": [
                "High",
                "Medium",
                "Low"
            ],

            "Predicted 90-Day Revenue": [
                3000,
                1500,
                500
            ],

            "CLV Value Band": [
                "High",
                "Medium",
                "Low"
            ],

            "Top Recommended Product": [
                "Product A",
                "Product B",
                "Product C"
            ],

            "Recommendation Source": [
                "Hybrid",
                "Hybrid",
                "Hybrid"
            ]
        }
    )

    path = os.path.join(
        tmp_path,
        "customers.csv"
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
        CustomerIntelligenceService
        .normalize_customer_id(
            101
        )
        ==
        "101"
    )

    assert (
        CustomerIntelligenceService
        .normalize_customer_id(
            101.0
        )
        ==
        "101"
    )

    assert (
        CustomerIntelligenceService
        .normalize_customer_id(
            "101.0"
        )
        ==
        "101"
    )


# ============================================================
# LOAD DATA
# ============================================================

def test_load_customer_data(
    tmp_path
):

    path = create_test_csv(
        tmp_path
    )

    service = (
        CustomerIntelligenceService(
            path
        )
    )

    result = (
        service.load_data()
    )

    assert (
        len(result)
        ==
        3
    )

    assert (
        "Customer ID"
        in result.columns
    )

    assert (
        result.iloc[0][
            "Customer ID"
        ]
        ==
        "101"
    )


# ============================================================
# GET CUSTOMER
# ============================================================

def test_get_customer(
    tmp_path
):

    path = create_test_csv(
        tmp_path
    )

    service = (
        CustomerIntelligenceService(
            path
        )
    )

    customer = (
        service.get_customer(
            "101"
        )
    )

    assert customer is not None

    assert (
        customer[
            "Customer ID"
        ]
        ==
        "101"
    )

    assert (
        customer[
            "Churn Risk"
        ]
        ==
        "High"
    )


# ============================================================
# CUSTOMER SUMMARY
# ============================================================

def test_customer_summary(
    tmp_path
):

    path = create_test_csv(
        tmp_path
    )

    service = (
        CustomerIntelligenceService(
            path
        )
    )

    summary = (
        service
        .get_customer_summary(
            "101"
        )
    )

    assert (
        summary[
            "customer_id"
        ]
        ==
        "101"
    )

    assert (
        summary[
            "churn_probability"
        ]
        ==
        0.8
    )

    assert (
        summary[
            "clv_value_band"
        ]
        ==
        "High"
    )


# ============================================================
# CUSTOMER NOT FOUND
# ============================================================

def test_customer_not_found(
    tmp_path
):

    path = create_test_csv(
        tmp_path
    )

    service = (
        CustomerIntelligenceService(
            path
        )
    )

    result = (
        service.get_customer(
            "999"
        )
    )

    assert result is None


# ============================================================
# LIST CUSTOMERS
# ============================================================

def test_list_customers(
    tmp_path
):

    path = create_test_csv(
        tmp_path
    )

    service = (
        CustomerIntelligenceService(
            path
        )
    )

    result = (
        service.list_customers(
            limit=2,
            offset=0
        )
    )

    assert (
        result[
            "total"
        ]
        ==
        3
    )

    assert (
        len(
            result[
                "customers"
            ]
        )
        ==
        2
    )


# ============================================================
# DATASET STATUS
# ============================================================

def test_dataset_status(
    tmp_path
):

    path = create_test_csv(
        tmp_path
    )

    service = (
        CustomerIntelligenceService(
            path
        )
    )

    result = (
        service.dataset_status()
    )

    assert (
        result[
            "available"
        ]
        is True
    )

    assert (
        result[
            "customers"
        ]
        ==
        3
    )