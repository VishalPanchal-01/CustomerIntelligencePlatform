import pandas as pd

from src.evaluation.recommendation_temporal_split import (
    RecommendationTemporalSplitter
)


def create_transaction_data():

    return pd.DataFrame(
        {
            "CustomerID": [
                101,
                101,
                101,
                102,
                102,
                103
            ],

            "Invoice": [
                "A1",
                "A2",
                "A3",
                "B1",
                "B2",
                "C1"
            ],

            "StockCode": [
                "P1",
                "P2",
                "P3",
                "P1",
                "P4",
                "P5"
            ],

            "Description": [
                "Product 1",
                "Product 2",
                "Product 3",
                "Product 1",
                "Product 4",
                "Product 5"
            ],

            "Quantity": [
                1,
                2,
                1,
                3,
                1,
                2
            ],

            "InvoiceDate": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-02-01",
                    "2025-03-01",
                    "2025-01-10",
                    "2025-03-10",
                    "2025-02-15"
                ]
            ),

            "UnitPrice": [
                10,
                20,
                30,
                10,
                15,
                25
            ],

            "Revenue": [
                10,
                40,
                30,
                30,
                15,
                50
            ]
        }
    )


def test_recommendation_temporal_split():

    df = (
        create_transaction_data()
    )

    splitter = (
        RecommendationTemporalSplitter(
            holdout_invoices=1,
            minimum_invoices=2
        )
    )

    (
        train,
        test,
        eligible_customers
    ) = splitter.split(
        df
    )

    # ---------------------------------
    # Customers 101 and 102 have >= 2
    # invoices and are evaluable.
    #
    # Customer 103 has only one invoice.
    # ---------------------------------

    assert set(
        eligible_customers
    ) == {
        101,
        102
    }

    # ---------------------------------
    # Latest invoice for customer 101
    # should be A3
    # ---------------------------------

    customer_101_test = (
        test[
            test[
                "CustomerID"
            ]
            ==
            101
        ]
    )

    assert (
        customer_101_test[
            "Invoice"
        ]
        .iloc[0]
        ==
        "A3"
    )

    # ---------------------------------
    # Latest invoice for customer 102
    # should be B2
    # ---------------------------------

    customer_102_test = (
        test[
            test[
                "CustomerID"
            ]
            ==
            102
        ]
    )

    assert (
        customer_102_test[
            "Invoice"
        ]
        .iloc[0]
        ==
        "B2"
    )

    # ---------------------------------
    # Customer 103 stays in training
    # because there is no evaluation
    # history available.
    # ---------------------------------

    assert (
        103
        in
        train[
            "CustomerID"
        ].values
    )

    # ---------------------------------
    # No invoice should exist in both
    # train and test.
    # ---------------------------------

    train_keys = set(
        zip(
            train[
                "CustomerID"
            ],
            train[
                "Invoice"
            ]
        )
    )

    test_keys = set(
        zip(
            test[
                "CustomerID"
            ],
            test[
                "Invoice"
            ]
        )
    )

    assert (
        train_keys
        .isdisjoint(
            test_keys
        )
    )