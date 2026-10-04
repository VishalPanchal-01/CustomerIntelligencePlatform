import os

import pandas as pd

from dashboard.ui import (
    get_dataset_information,
    dataframe_to_csv_bytes
)


# ============================================================
# DATASET INFO
# ============================================================

def test_dataset_information(
    tmp_path
):

    df = pd.DataFrame(
        {
            "Customer ID": [
                101,
                102,
                103
            ],

            "Customer Segment": [
                "VIP",
                "Regular",
                "At Risk"
            ]
        }
    )

    file_path = os.path.join(
        tmp_path,
        "customers.csv"
    )

    df.to_csv(
        file_path,
        index=False
    )

    result = (
        get_dataset_information(
            df,
            file_path
        )
    )

    assert (
        result[
            "rows"
        ]
        ==
        3
    )

    assert (
        result[
            "columns"
        ]
        ==
        2
    )

    assert (
        result[
            "customers"
        ]
        ==
        3
    )

    assert (
        result[
            "file_size_mb"
        ]
        >=
        0
    )

    assert (
        result[
            "last_updated"
        ]
        !=
        "Unknown"
    )


# ============================================================
# CSV EXPORT
# ============================================================

def test_dataframe_to_csv_bytes():

    df = pd.DataFrame(
        {
            "Customer ID": [
                101,
                102
            ],

            "Churn Risk": [
                "Low",
                "High"
            ]
        }
    )

    result = (
        dataframe_to_csv_bytes(
            df
        )
    )

    assert isinstance(
        result,
        bytes
    )

    text = result.decode(
        "utf-8"
    )

    assert (
        "Customer ID"
        in text
    )

    assert (
        "Churn Risk"
        in text
    )

    assert (
        "101"
        in text
    )