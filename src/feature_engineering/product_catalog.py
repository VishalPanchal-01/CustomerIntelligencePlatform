import sys

import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class ProductCatalogBuilder:

    def build_catalog(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Starting product catalog creation."
            )

            required_columns = [
                "StockCode",
                "Description",
                "UnitPrice",
                "InvoiceDate"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing product catalog columns: "
                    f"{missing_columns}"
                )

            if df.empty:

                raise ValueError(
                    "Product source data is empty."
                )

            data = df.copy()

            # ---------------------------------
            # Sort so latest description/price
            # becomes available
            # ---------------------------------

            data = data.sort_values(
                by="InvoiceDate"
            )

            catalog = (
                data
                .groupby(
                    "StockCode",
                    as_index=False
                )
                .agg(
                    Description=(
                        "Description",
                        "last"
                    ),

                    LatestUnitPrice=(
                        "UnitPrice",
                        "last"
                    ),

                    TotalCustomers=(
                        "CustomerID",
                        "nunique"
                    ),

                    TotalPurchases=(
                        "Invoice",
                        "nunique"
                    ),

                    TotalQuantity=(
                        "Quantity",
                        "sum"
                    )
                )
            )

            catalog = (
                catalog
                .sort_values(
                    by="TotalCustomers",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )

            logger.info(
                f"Product catalog created. "
                f"Products: {len(catalog)}"
            )

            return catalog

        except Exception as e:

            logger.error(
                "Product catalog creation failed."
            )

            raise CustomException(
                e,
                sys
            )