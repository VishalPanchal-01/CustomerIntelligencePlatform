import pandas as pd

from src.feature_engineering.product_catalog import (
    ProductCatalogBuilder
)


def test_product_catalog():

    df = pd.DataFrame(
        {
            "CustomerID": [
                101,
                102,
                103
            ],

            "Invoice": [
                "A1",
                "A2",
                "A3"
            ],

            "StockCode": [
                "P1",
                "P1",
                "P2"
            ],

            "Description": [
                "Old Product Name",
                "New Product Name",
                "Product Two"
            ],

            "Quantity": [
                2,
                3,
                4
            ],

            "InvoiceDate": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-02-01",
                    "2025-01-15"
                ]
            ),

            "UnitPrice": [
                10,
                12,
                20
            ]
        }
    )

    builder = (
        ProductCatalogBuilder()
    )

    result = (
        builder.build_catalog(
            df
        )
    )

    assert result is not None

    assert (
        result[
            "StockCode"
        ]
        .nunique()
        == 2
    )

    product_p1 = (
        result[
            result[
                "StockCode"
            ]
            ==
            "P1"
        ]
        .iloc[0]
    )

    assert (
        product_p1[
            "Description"
        ]
        ==
        "New Product Name"
    )

    assert (
        product_p1[
            "LatestUnitPrice"
        ]
        ==
        12
    )

    assert (
        product_p1[
            "TotalCustomers"
        ]
        ==
        2
    )