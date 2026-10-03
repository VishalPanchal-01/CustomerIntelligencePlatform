import sys

import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class RecommendationDataCleaning:

    def clean_purchase_transactions(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Starting recommendation transaction cleaning."
            )

            data = df.copy()

            # ---------------------------------
            # Handle common column names
            # ---------------------------------

            rename_mapping = {}

            if (
                "Customer ID" in data.columns
                and
                "CustomerID" not in data.columns
            ):

                rename_mapping[
                    "Customer ID"
                ] = "CustomerID"

            if (
                "InvoiceNo" in data.columns
                and
                "Invoice" not in data.columns
            ):

                rename_mapping[
                    "InvoiceNo"
                ] = "Invoice"

            if (
                "Price" in data.columns
                and
                "UnitPrice" not in data.columns
            ):

                rename_mapping[
                    "Price"
                ] = "UnitPrice"

            if rename_mapping:

                data = data.rename(
                    columns=rename_mapping
                )

            # ---------------------------------
            # Validate required columns
            # ---------------------------------

            required_columns = [
                "CustomerID",
                "Invoice",
                "StockCode",
                "Description",
                "Quantity",
                "InvoiceDate",
                "UnitPrice"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in data.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing recommendation columns: "
                    f"{missing_columns}"
                )

            # ---------------------------------
            # Date conversion
            # ---------------------------------

            data[
                "InvoiceDate"
            ] = pd.to_datetime(
                data["InvoiceDate"],
                errors="coerce"
            )

            # ---------------------------------
            # Numeric conversion
            # ---------------------------------

            data[
                "Quantity"
            ] = pd.to_numeric(
                data["Quantity"],
                errors="coerce"
            )

            data[
                "UnitPrice"
            ] = pd.to_numeric(
                data["UnitPrice"],
                errors="coerce"
            )

            # ---------------------------------
            # Remove unusable rows
            # ---------------------------------

            data = data.dropna(
                subset=[
                    "CustomerID",
                    "Invoice",
                    "StockCode",
                    "Description",
                    "Quantity",
                    "InvoiceDate",
                    "UnitPrice"
                ]
            )

            # ---------------------------------
            # Convert identifiers to strings
            # ---------------------------------

            data[
                "Invoice"
            ] = (
                data[
                    "Invoice"
                ]
                .astype(str)
                .str.strip()
            )

            data[
                "StockCode"
            ] = (
                data[
                    "StockCode"
                ]
                .astype(str)
                .str.strip()
            )

            data[
                "Description"
            ] = (
                data[
                    "Description"
                ]
                .astype(str)
                .str.strip()
            )

            # ---------------------------------
            # Remove cancellation invoices
            # ---------------------------------

            cancellation_mask = (
                data[
                    "Invoice"
                ]
                .str.upper()
                .str.startswith(
                    "C"
                )
            )

            data = data[
                ~cancellation_mask
            ].copy()

            # ---------------------------------
            # Keep positive purchases only
            # ---------------------------------

            data = data[
                (
                    data[
                        "Quantity"
                    ]
                    > 0
                )
                &
                (
                    data[
                        "UnitPrice"
                    ]
                    > 0
                )
            ].copy()

            # ---------------------------------
            # Remove empty product identifiers
            # ---------------------------------

            data = data[
                data[
                    "StockCode"
                ]
                .ne("")
            ].copy()

            data = data[
                data[
                    "Description"
                ]
                .ne("")
            ].copy()

            # ---------------------------------
            # Calculate revenue
            # ---------------------------------

            data[
                "Revenue"
            ] = (
                data[
                    "Quantity"
                ]
                *
                data[
                    "UnitPrice"
                ]
            )

            # ---------------------------------
            # Final validation
            # ---------------------------------

            if data.empty:

                raise ValueError(
                    "No valid recommendation "
                    "transactions remain after cleaning."
                )

            data = data.reset_index(
                drop=True
            )

            logger.info(
                f"Recommendation transactions cleaned. "
                f"Shape: {data.shape}"
            )

            return data

        except Exception as e:

            logger.error(
                "Recommendation transaction cleaning failed."
            )

            raise CustomException(
                e,
                sys
            )