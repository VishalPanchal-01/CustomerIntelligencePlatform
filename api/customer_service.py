import os

import numpy as np
import pandas as pd


CUSTOMER_ID = "Customer ID"


class CustomerIntelligenceService:

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        data_path: str
    ):

        self.data_path = data_path

        self.data = pd.DataFrame()


    # =========================================================
    # NORMALIZE CUSTOMER ID
    # =========================================================

    @staticmethod
    def normalize_customer_id(
        value
    ) -> str:

        if pd.isna(
            value
        ):

            return ""

        text = str(
            value
        ).strip()

        try:

            numeric = float(
                text
            )

            if numeric.is_integer():

                return str(
                    int(
                        numeric
                    )
                )

        except (
            TypeError,
            ValueError
        ):

            pass

        return text


    # =========================================================
    # NORMALIZE COLUMN
    # =========================================================

    def _normalize_customer_column(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = df.copy()

        # -----------------------------------------------------
        # Backward compatibility
        # -----------------------------------------------------

        if (
            "CustomerID" in data.columns
            and
            CUSTOMER_ID not in data.columns
        ):

            data = data.rename(
                columns={
                    "CustomerID":
                        CUSTOMER_ID
                }
            )

        if CUSTOMER_ID not in data.columns:

            raise ValueError(
                "Customer ID column not found "
                "in unified customer dataset."
            )

        data[
            CUSTOMER_ID
        ] = data[
            CUSTOMER_ID
        ].apply(
            self.normalize_customer_id
        )

        return data


    # =========================================================
    # LOAD DATA
    # =========================================================

    def load_data(
        self
    ) -> pd.DataFrame:

        if not os.path.exists(
            self.data_path
        ):

            raise FileNotFoundError(
                "Unified customer intelligence "
                f"dataset not found: {self.data_path}"
            )

        data = pd.read_csv(
            self.data_path
        )

        data = (
            self._normalize_customer_column(
                data
            )
        )

        # Remove invalid customer IDs
        data = (
            data[
                data[
                    CUSTOMER_ID
                ]
                !=
                ""
            ]
            .copy()
        )

        # Keep one row per customer
        data = (
            data
            .drop_duplicates(
                subset=[
                    CUSTOMER_ID
                ],
                keep="first"
            )
            .reset_index(
                drop=True
            )
        )

        self.data = data

        return self.data


    # =========================================================
    # ENSURE DATA LOADED
    # =========================================================

    def _ensure_loaded(
        self
    ) -> None:

        if self.data.empty:

            self.load_data()


    # =========================================================
    # CONVERT VALUE TO JSON-SAFE
    # =========================================================

    @staticmethod
    def json_safe_value(
        value
    ):

        if pd.isna(
            value
        ):

            return None

        if isinstance(
            value,
            (
                np.integer,
                np.int8,
                np.int16,
                np.int32,
                np.int64
            )
        ):

            return int(
                value
            )

        if isinstance(
            value,
            (
                np.floating,
                np.float16,
                np.float32,
                np.float64
            )
        ):

            return float(
                value
            )

        if isinstance(
            value,
            np.bool_
        ):

            return bool(
                value
            )

        if isinstance(
            value,
            pd.Timestamp
        ):

            return value.isoformat()

        return value


    # =========================================================
    # GET CUSTOMER
    # =========================================================

    def get_customer(
        self,
        customer_id
    ) -> dict | None:

        self._ensure_loaded()

        normalized_id = (
            self.normalize_customer_id(
                customer_id
            )
        )

        match = (
            self.data[
                self.data[
                    CUSTOMER_ID
                ]
                ==
                normalized_id
            ]
        )

        if match.empty:

            return None

        row = (
            match.iloc[0]
        )

        result = {}

        for column, value in row.items():

            result[
                column
            ] = self.json_safe_value(
                value
            )

        return result


    # =========================================================
    # GET CUSTOMER SUMMARY
    # =========================================================

    def get_customer_summary(
        self,
        customer_id
    ) -> dict | None:

        customer = self.get_customer(
            customer_id
        )

        if customer is None:

            return None

        return {

            "customer_id":
                self.normalize_customer_id(
                    customer.get(
                        CUSTOMER_ID
                    )
                ),

            "segment":
                customer.get(
                    "Customer Segment"
                ),

            "churn_probability":
                customer.get(
                    "Churn Probability"
                ),

            "churn_risk":
                customer.get(
                    "Churn Risk"
                ),

            "predicted_90_day_revenue":
                customer.get(
                    "Predicted 90-Day Revenue"
                ),

            "clv_value_band":
                customer.get(
                    "CLV Value Band"
                ),

            "top_recommended_product":
                customer.get(
                    "Top Recommended Product"
                ),

            "recommendation_source":
                customer.get(
                    "Recommendation Source"
                )
        }


    # =========================================================
    # LIST CUSTOMERS
    # =========================================================

    def list_customers(
        self,
        limit: int = 20,
        offset: int = 0
    ) -> dict:

        self._ensure_loaded()

        subset = (
            self.data
            .iloc[
                offset:
                offset + limit
            ]
            .copy()
        )

        customers = []

        for _, row in subset.iterrows():

            customers.append(
                {
                    "customer_id":
                        self.normalize_customer_id(
                            row.get(
                                CUSTOMER_ID
                            )
                        ),

                    "segment":
                        self.json_safe_value(
                            row.get(
                                "Customer Segment"
                            )
                        ),

                    "churn_risk":
                        self.json_safe_value(
                            row.get(
                                "Churn Risk"
                            )
                        ),

                    "clv_value_band":
                        self.json_safe_value(
                            row.get(
                                "CLV Value Band"
                            )
                        )
                }
            )

        return {

            "total":
                int(
                    len(
                        self.data
                    )
                ),

            "limit":
                limit,

            "offset":
                offset,

            "customers":
                customers
        }


    # =========================================================
    # DATASET STATUS
    # =========================================================

    def dataset_status(
        self
    ) -> dict:

        exists = os.path.exists(
            self.data_path
        )

        if not exists:

            return {

                "available":
                    False,

                "rows":
                    0,

                "customers":
                    0,

                "columns":
                    0,

                "path":
                    self.data_path
            }

        self._ensure_loaded()

        return {

            "available":
                True,

            "rows":
                int(
                    len(
                        self.data
                    )
                ),

            "customers":
                int(
                    self.data[
                        CUSTOMER_ID
                    ]
                    .nunique()
                ),

            "columns":
                int(
                    len(
                        self.data.columns
                    )
                ),

            "path":
                self.data_path
        }