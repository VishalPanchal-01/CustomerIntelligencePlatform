import json
import os

import pandas as pd


CUSTOMER_ID = "Customer ID"


class ExplainabilityValidation:

    # =========================================================
    # FILE CHECK
    # =========================================================

    @staticmethod
    def check_file(
        path: str
    ) -> dict:

        exists = os.path.exists(
            path
        )

        return {
            "path":
                path,

            "exists":
                exists,

            "size_bytes":
                (
                    os.path.getsize(
                        path
                    )
                    if exists
                    else 0
                )
        }


    # =========================================================
    # LOAD CSV
    # =========================================================

    @staticmethod
    def load_csv(
        path: str
    ) -> pd.DataFrame:

        if not os.path.exists(
            path
        ):

            return pd.DataFrame()

        return pd.read_csv(
            path
        )


    # =========================================================
    # LOAD JSON
    # =========================================================

    @staticmethod
    def load_json(
        path: str
    ) -> dict:

        if not os.path.exists(
            path
        ):

            return {}

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(
                file
            )


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
    # NORMALIZE CUSTOMER COLUMN
    # =========================================================

    def normalize_customer_column(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:

        data = df.copy()

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

        if CUSTOMER_ID in data.columns:

            data[
                CUSTOMER_ID
            ] = data[
                CUSTOMER_ID
            ].apply(
                self.normalize_customer_id
            )

        return data


    # =========================================================
    # VALIDATE IMPORTANCE DATA
    # =========================================================

    @staticmethod
    def validate_importance_data(
        df: pd.DataFrame
    ) -> dict:

        required_columns = [
            "Feature",
            "Mean Absolute SHAP"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        valid = (
            not df.empty
            and
            not missing_columns
        )

        return {
            "valid":
                valid,

            "rows":
                int(
                    len(
                        df
                    )
                ),

            "missing_columns":
                missing_columns
        }


    # =========================================================
    # VALIDATE LOCAL SHAP DATA
    # =========================================================

    def validate_local_shap_data(
        self,
        df: pd.DataFrame
    ) -> dict:

        data = self.normalize_customer_column(
            df
        )

        required_columns = [
            CUSTOMER_ID,
            "Feature",
            "Feature Value",
            "SHAP Value",
            "Absolute SHAP"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in data.columns
        ]

        valid = (
            not data.empty
            and
            not missing_columns
        )

        if (
            valid
            and
            CUSTOMER_ID in data.columns
        ):

            explained_customers = int(
                data[
                    CUSTOMER_ID
                ]
                .nunique()
            )

        else:

            explained_customers = 0

        return {
            "valid":
                valid,

            "rows":
                int(
                    len(
                        data
                    )
                ),

            "explained_customers":
                explained_customers,

            "missing_columns":
                missing_columns
        }


    # =========================================================
    # TOP FEATURES
    # =========================================================

    @staticmethod
    def top_features(
        importance_df: pd.DataFrame,
        top_n: int = 6
    ) -> list:

        if (
            importance_df.empty
            or
            "Feature"
            not in importance_df.columns
            or
            "Mean Absolute SHAP"
            not in importance_df.columns
        ):

            return []

        data = importance_df.copy()

        data[
            "Mean Absolute SHAP"
        ] = pd.to_numeric(
            data[
                "Mean Absolute SHAP"
            ],
            errors="coerce"
        )

        data = (
            data
            .dropna(
                subset=[
                    "Mean Absolute SHAP"
                ]
            )
            .sort_values(
                by=
                    "Mean Absolute SHAP",
                ascending=False
            )
            .head(
                top_n
            )
        )

        return (
            data[
                [
                    "Feature",
                    "Mean Absolute SHAP"
                ]
            ]
            .to_dict(
                orient="records"
            )
        )


    # =========================================================
    # EXPLANATION COVERAGE
    # =========================================================

    def explanation_coverage(
        self,
        local_shap_df: pd.DataFrame,
        total_customer_df: pd.DataFrame
    ) -> dict:

        shap_data = (
            self.normalize_customer_column(
                local_shap_df
            )
        )

        customer_data = (
            self.normalize_customer_column(
                total_customer_df
            )
        )

        if (
            CUSTOMER_ID not in shap_data.columns
            or
            CUSTOMER_ID not in customer_data.columns
        ):

            return {
                "explained_customers":
                    0,

                "total_customers":
                    0,

                "coverage_percentage":
                    0.0
            }

        explained_customers = int(
            shap_data[
                CUSTOMER_ID
            ]
            .nunique()
        )

        total_customers = int(
            customer_data[
                CUSTOMER_ID
            ]
            .nunique()
        )

        if total_customers == 0:

            coverage = 0.0

        else:

            coverage = (
                explained_customers
                /
                total_customers
                *
                100
            )

        return {
            "explained_customers":
                explained_customers,

            "total_customers":
                total_customers,

            "coverage_percentage":
                float(
                    coverage
                )
        }


    # =========================================================
    # COMMON EXPLAINED CUSTOMERS
    # =========================================================

    def common_explained_customers(
        self,
        churn_shap_df: pd.DataFrame,
        clv_shap_df: pd.DataFrame
    ) -> dict:

        churn = self.normalize_customer_column(
            churn_shap_df
        )

        clv = self.normalize_customer_column(
            clv_shap_df
        )

        if (
            CUSTOMER_ID not in churn.columns
            or
            CUSTOMER_ID not in clv.columns
        ):

            return {
                "count":
                    0,

                "customers":
                    []
            }

        churn_customers = set(
            churn[
                CUSTOMER_ID
            ]
            .dropna()
            .astype(str)
        )

        clv_customers = set(
            clv[
                CUSTOMER_ID
            ]
            .dropna()
            .astype(str)
        )

        common = sorted(
            churn_customers
            &
            clv_customers
        )

        return {
            "count":
                len(
                    common
                ),

            "customers":
                common
        }