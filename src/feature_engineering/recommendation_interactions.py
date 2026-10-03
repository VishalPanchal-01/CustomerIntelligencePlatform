import sys

import numpy as np
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class RecommendationInteractionBuilder:

    def build_interactions(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        try:

            logger.info(
                "Starting customer-product interaction creation."
            )

            required_columns = [
                "CustomerID",
                "Invoice",
                "StockCode",
                "Description",
                "Quantity",
                "InvoiceDate",
                "Revenue"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:

                raise ValueError(
                    f"Missing interaction columns: "
                    f"{missing_columns}"
                )

            if df.empty:

                raise ValueError(
                    "Recommendation transaction dataset is empty."
                )

            data = df.copy()

            # ---------------------------------
            # Customer-product aggregation
            # ---------------------------------

            interactions = (
                data
                .groupby(
                    [
                        "CustomerID",
                        "StockCode"
                    ],
                    as_index=False
                )
                .agg(
                    Description=(
                        "Description",
                        "first"
                    ),

                    PurchaseCount=(
                        "Invoice",
                        "nunique"
                    ),

                    TotalQuantity=(
                        "Quantity",
                        "sum"
                    ),

                    TotalSpend=(
                        "Revenue",
                        "sum"
                    ),

                    LastPurchaseDate=(
                        "InvoiceDate",
                        "max"
                    )
                )
            )

            # ---------------------------------
            # Interaction strength
            # ---------------------------------
            #
            # We use:
            #
            # log1p(PurchaseCount)
            # +
            # log1p(TotalQuantity)
            #
            # This rewards both:
            # - repeat purchasing
            # - purchase volume
            #
            # while reducing domination by
            # extremely large quantities.
            # ---------------------------------

            interactions[
                "InteractionStrength"
            ] = (
                np.log1p(
                    interactions[
                        "PurchaseCount"
                    ]
                )
                +
                np.log1p(
                    interactions[
                        "TotalQuantity"
                    ]
                )
            )

            # ---------------------------------
            # Validation
            # ---------------------------------

            if (
                interactions[
                    "PurchaseCount"
                ]
                <= 0
            ).any():

                raise ValueError(
                    "PurchaseCount must be positive."
                )

            if (
                interactions[
                    "TotalQuantity"
                ]
                <= 0
            ).any():

                raise ValueError(
                    "TotalQuantity must be positive."
                )

            if (
                interactions[
                    "TotalSpend"
                ]
                <= 0
            ).any():

                raise ValueError(
                    "TotalSpend must be positive."
                )

            if (
                interactions[
                    "InteractionStrength"
                ]
                <= 0
            ).any():

                raise ValueError(
                    "InteractionStrength must be positive."
                )

            # ---------------------------------
            # Sort
            # ---------------------------------

            interactions = (
                interactions
                .sort_values(
                    by=[
                        "CustomerID",
                        "InteractionStrength"
                    ],
                    ascending=[
                        True,
                        False
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            logger.info(
                f"Customer-product interaction "
                f"dataset created. "
                f"Shape: {interactions.shape}"
            )

            return interactions

        except Exception as e:

            logger.error(
                "Recommendation interaction creation failed."
            )

            raise CustomException(
                e,
                sys
            )