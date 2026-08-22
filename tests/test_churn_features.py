import os

import pandas as pd

from src.feature_engineering.churn_features import (
    ChurnFeatureEngineer
)


def create_test_data():

    return pd.DataFrame(
        {
            "Customer ID": [
                101,
                101,
                102,
                102,
                103,
                103
            ],

            "Invoice": [
                "INV001",
                "INV002",
                "INV003",
                "INV004",
                "INV005",
                "INV006"
            ],

            "InvoiceDate": pd.to_datetime(
                [
                    "2011-08-01",
                    "2011-08-15",
                    "2011-08-05",
                    "2011-08-20",
                    "2011-08-10",
                    "2011-08-25"
                ]
            ),

            "Revenue": [
                100,
                200,
                150,
                250,
                300,
                100
            ]
        }
    )


def test_create_customer_features():

    engineer = ChurnFeatureEngineer()

    df = create_test_data()

    observation_date = pd.Timestamp(
        "2011-09-01"
    )

    result = engineer.create_customer_features(
        df,
        observation_date
    )

    assert result is not None

    assert len(result) == 3

    expected_columns = [
        "Customer ID",
        "Recency",
        "Frequency",
        "Monetary",
        "AverageOrderValue",
        "Tenure"
    ]

    for column in expected_columns:
        assert column in result.columns


def test_customer_feature_values():

    engineer = ChurnFeatureEngineer()

    df = create_test_data()

    observation_date = pd.Timestamp(
        "2011-09-01"
    )

    result = engineer.create_customer_features(
        df,
        observation_date
    )

    customer_101 = result[
        result["Customer ID"] == 101
    ].iloc[0]

    assert customer_101["Frequency"] == 2

    assert customer_101["Monetary"] == 300

    assert customer_101["AverageOrderValue"] == 150

    assert customer_101["Recency"] == 17


def test_create_churn_label():

    engineer = ChurnFeatureEngineer()

    df = create_test_data()

    # Add a future transaction for customer 101
    future_transaction = pd.DataFrame(
        {
            "Customer ID": [101],
            "InvoiceNo": ["INV007"],
            "InvoiceDate": pd.to_datetime(
                ["2011-09-15"]
            ),
            "Revenue": [500]
        }
    )

    df = pd.concat(
        [
            df,
            future_transaction
        ],
        ignore_index=True
    )

    observation_date = pd.Timestamp(
        "2011-09-01"
    )

    customer_features = (
        engineer.create_customer_features(
            df[
                df["InvoiceDate"]
                <= observation_date
            ],
            observation_date
        )
    )

    result = engineer.create_churn_label(
        df,
        customer_features,
        observation_date,
        churn_window_days=90
    )

    customer_101 = result[
        result["Customer ID"] == 101
    ].iloc[0]

    customer_102 = result[
        result["Customer ID"] == 102
    ].iloc[0]

    assert customer_101["Churn"] == 0

    assert customer_102["Churn"] == 1


def test_build_churn_dataset():

    engineer = ChurnFeatureEngineer()

    df = create_test_data()

    observation_date = pd.Timestamp(
        "2011-09-01"
    )

    result = engineer.build_churn_dataset(
        df,
        observation_date
    )

    assert result is not None

    assert "Churn" in result.columns

    assert len(result) == 3

    output_path = os.path.join(
        engineer.output_dir,
        "customer_churn_dataset.csv"
    )

    assert os.path.exists(output_path)

    assert os.path.getsize(output_path) > 0