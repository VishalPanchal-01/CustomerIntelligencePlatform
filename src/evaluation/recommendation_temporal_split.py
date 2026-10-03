import sys

import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class RecommendationTemporalSplitter:

    def __init__(
        self,
        holdout_invoices: int = 1,
        minimum_invoices: int = 2
    ):

        self.holdout_invoices = (
            holdout_invoices
        )

        self.minimum_invoices = (
            minimum_invoices
        )

    def split(
        self,
        df: pd.DataFrame
    ) -> tuple:

        try:

            logger.info(
                "Starting temporal recommendation split."
            )

            # ---------------------------------
            # Validate configuration
            # ---------------------------------

            if self.holdout_invoices < 1:

                raise ValueError(
                    "holdout_invoices must be at least 1."
                )

            if self.minimum_invoices < 2:

                raise ValueError(
                    "minimum_invoices must be at least 2."
                )

            if (
                self.minimum_invoices
                <=
                self.holdout_invoices
            ):

                raise ValueError(
                    "minimum_invoices must be greater "
                    "than holdout_invoices."
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
                "UnitPrice",
                "Revenue"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing recommendation split columns: "
                    f"{missing_columns}"
                )

            if df.empty:

                raise ValueError(
                    "Recommendation transaction dataset is empty."
                )

            data = df.copy()

            # ---------------------------------
            # Prepare dates and identifiers
            # ---------------------------------

            data[
                "InvoiceDate"
            ] = pd.to_datetime(
                data["InvoiceDate"],
                errors="coerce"
            )

            data = data.dropna(
                subset=[
                    "CustomerID",
                    "Invoice",
                    "StockCode",
                    "InvoiceDate"
                ]
            )

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

            # ---------------------------------
            # Build one row per invoice
            # ---------------------------------

            invoice_history = (
                data
                .groupby(
                    [
                        "CustomerID",
                        "Invoice"
                    ],
                    as_index=False
                )
                .agg(
                    InvoiceDate=(
                        "InvoiceDate",
                        "max"
                    )
                )
            )

            # ---------------------------------
            # Number of orders per customer
            # ---------------------------------

            invoice_counts = (
                invoice_history
                .groupby(
                    "CustomerID"
                )[
                    "Invoice"
                ]
                .nunique()
            )

            eligible_customers = (
                invoice_counts[
                    invoice_counts
                    >=
                    self.minimum_invoices
                ]
                .index
            )

            logger.info(
                f"Customers eligible for temporal "
                f"evaluation: {len(eligible_customers)}"
            )

            # ---------------------------------
            # Get invoices belonging only to
            # evaluation-eligible customers
            # ---------------------------------

            eligible_invoice_history = (
                invoice_history[
                    invoice_history[
                        "CustomerID"
                    ]
                    .isin(
                        eligible_customers
                    )
                ]
                .copy()
            )

            # ---------------------------------
            # Sort invoices chronologically
            #
            # Invoice is used as secondary key
            # for deterministic ordering when
            # timestamps are equal.
            # ---------------------------------

            eligible_invoice_history = (
                eligible_invoice_history
                .sort_values(
                    by=[
                        "CustomerID",
                        "InvoiceDate",
                        "Invoice"
                    ]
                )
            )

            # ---------------------------------
            # Rank invoices from latest backwards
            #
            # Latest invoice => rank 1
            # ---------------------------------

            eligible_invoice_history[
                "ReverseInvoiceRank"
            ] = (
                eligible_invoice_history
                .groupby(
                    "CustomerID"
                )
                .cumcount(
                    ascending=False
                )
                +
                1
            )

            # ---------------------------------
            # Select latest N invoices
            # ---------------------------------

            heldout_invoice_keys = (
                eligible_invoice_history[
                    eligible_invoice_history[
                        "ReverseInvoiceRank"
                    ]
                    <=
                    self.holdout_invoices
                ][
                    [
                        "CustomerID",
                        "Invoice"
                    ]
                ]
                .copy()
            )

            heldout_invoice_keys[
                "IsHoldout"
            ] = True

            # ---------------------------------
            # Mark original transaction rows
            # ---------------------------------

            marked_data = (
                data
                .merge(
                    heldout_invoice_keys,
                    on=[
                        "CustomerID",
                        "Invoice"
                    ],
                    how="left"
                )
            )

            marked_data[
                "IsHoldout"
            ] = (
                marked_data[
                    "IsHoldout"
                ]
                .fillna(False)
                .astype(bool)
            )

            # ---------------------------------
            # Training transactions
            #
            # Important:
            # Customers with only one invoice
            # remain entirely in training.
            #
            # They are not used for evaluation,
            # but their interactions can still
            # help train the recommender.
            # ---------------------------------

            train_transactions = (
                marked_data[
                    ~marked_data[
                        "IsHoldout"
                    ]
                ]
                .drop(
                    columns=[
                        "IsHoldout"
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            # ---------------------------------
            # Held-out transactions
            # ---------------------------------

            test_transactions = (
                marked_data[
                    marked_data[
                        "IsHoldout"
                    ]
                ]
                .drop(
                    columns=[
                        "IsHoldout"
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            # ---------------------------------
            # Final safety validation
            # ---------------------------------

            if train_transactions.empty:

                raise ValueError(
                    "Training recommendation transactions "
                    "are empty after temporal split."
                )

            if test_transactions.empty:

                raise ValueError(
                    "No held-out recommendation "
                    "transactions were created."
                )

            # ---------------------------------
            # Leakage check
            #
            # Exact CustomerID + Invoice pair
            # must not exist in both.
            # ---------------------------------

            train_invoice_keys = set(
                zip(
                    train_transactions[
                        "CustomerID"
                    ],
                    train_transactions[
                        "Invoice"
                    ]
                )
            )

            test_invoice_keys = set(
                zip(
                    test_transactions[
                        "CustomerID"
                    ],
                    test_transactions[
                        "Invoice"
                    ]
                )
            )

            overlapping_invoices = (
                train_invoice_keys
                .intersection(
                    test_invoice_keys
                )
            )

            if overlapping_invoices:

                raise ValueError(
                    "Temporal split leakage detected: "
                    "some invoices exist in both train and test."
                )

            logger.info(
                f"Training transaction rows: "
                f"{len(train_transactions)}"
            )

            logger.info(
                f"Held-out transaction rows: "
                f"{len(test_transactions)}"
            )

            return (
                train_transactions,
                test_transactions,
                list(
                    eligible_customers
                )
            )

        except Exception as e:

            logger.error(
                "Temporal recommendation split failed."
            )

            raise CustomException(
                e,
                sys
            )