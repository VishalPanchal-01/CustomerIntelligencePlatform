import sys

import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class RecommendationGroundTruthBuilder:

    def build_ground_truth(
        self,
        test_transactions: pd.DataFrame,
        train_transactions: pd.DataFrame
    ) -> tuple:

        try:

            logger.info(
                "Building recommendation ground truth."
            )

            required_test_columns = [
                "CustomerID",
                "StockCode",
                "Description",
                "Invoice",
                "InvoiceDate"
            ]

            missing_test_columns = [
                column
                for column in required_test_columns
                if column not in test_transactions.columns
            ]

            if missing_test_columns:

                raise ValueError(
                    f"Missing test transaction columns: "
                    f"{missing_test_columns}"
                )

            if test_transactions.empty:

                raise ValueError(
                    "Test transactions are empty."
                )

            if train_transactions.empty:

                raise ValueError(
                    "Training transactions are empty."
                )

            # ---------------------------------
            # Products known during training
            # ---------------------------------

            train_products = set(
                train_transactions[
                    "StockCode"
                ]
                .astype(str)
                .unique()
            )

            # ---------------------------------
            # One row per customer-product
            # in held-out purchases
            # ---------------------------------

            ground_truth = (
                test_transactions[
                    [
                        "CustomerID",
                        "StockCode",
                        "Description",
                        "Invoice",
                        "InvoiceDate"
                    ]
                ]
                .drop_duplicates(
                    subset=[
                        "CustomerID",
                        "StockCode"
                    ]
                )
                .copy()
            )

            ground_truth[
                "StockCode"
            ] = (
                ground_truth[
                    "StockCode"
                ]
                .astype(str)
            )

            # ---------------------------------
            # Warm-item evaluation flag
            # ---------------------------------

            ground_truth[
                "IsEvaluableItem"
            ] = (
                ground_truth[
                    "StockCode"
                ]
                .isin(
                    train_products
                )
            )

            # ---------------------------------
            # Ground truth limited to products
            # the model could have learned
            # ---------------------------------

            evaluable_ground_truth = (
                ground_truth[
                    ground_truth[
                        "IsEvaluableItem"
                    ]
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            ground_truth = (
                ground_truth
                .reset_index(
                    drop=True
                )
            )

            logger.info(
                f"All held-out product interactions: "
                f"{len(ground_truth)}"
            )

            logger.info(
                f"Evaluable held-out interactions: "
                f"{len(evaluable_ground_truth)}"
            )

            return (
                ground_truth,
                evaluable_ground_truth
            )

        except Exception as e:

            logger.error(
                "Recommendation ground-truth "
                "creation failed."
            )

            raise CustomException(
                e,
                sys
            )