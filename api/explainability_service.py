import os

import pandas as pd


CUSTOMER_ID = "Customer ID"


class ExplainabilityService:

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        churn_global_path: str,
        churn_local_path: str,
        clv_global_path: str,
        clv_local_path: str
    ):

        self.churn_global_path = (
            churn_global_path
        )

        self.churn_local_path = (
            churn_local_path
        )

        self.clv_global_path = (
            clv_global_path
        )

        self.clv_local_path = (
            clv_local_path
        )

        self.churn_global = (
            pd.DataFrame()
        )

        self.churn_local = (
            pd.DataFrame()
        )

        self.clv_global = (
            pd.DataFrame()
        )

        self.clv_local = (
            pd.DataFrame()
        )


    # =========================================================
    # CUSTOMER ID NORMALIZATION
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
    # FILE STATUS
    # =========================================================

    @staticmethod
    def file_available(
        path: str
    ) -> bool:

        return os.path.exists(
            path
        )


    # =========================================================
    # LOAD GLOBAL DATA
    # =========================================================

    @staticmethod
    def _load_global(
        path: str
    ) -> pd.DataFrame:

        if not os.path.exists(
            path
        ):

            return pd.DataFrame()

        data = pd.read_csv(
            path
        )

        required = [
            "Feature",
            "Mean Absolute SHAP"
        ]

        missing = [
            column
            for column in required
            if column not in data.columns
        ]

        if missing:

            raise ValueError(
                "Invalid global SHAP file. "
                f"Missing columns: {missing}"
            )

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
            .reset_index(
                drop=True
            )
        )

        if "Importance Rank" not in data.columns:

            data[
                "Importance Rank"
            ] = (
                range(
                    1,
                    len(data) + 1
                )
            )

        return data


    # =========================================================
    # LOAD LOCAL DATA
    # =========================================================

    def _load_local(
        self,
        path: str
    ) -> pd.DataFrame:

        if not os.path.exists(
            path
        ):

            return pd.DataFrame()

        data = pd.read_csv(
            path
        )

        if (
            "CustomerID"
            in data.columns
            and
            CUSTOMER_ID
            not in data.columns
        ):

            data = data.rename(
                columns={
                    "CustomerID":
                        CUSTOMER_ID
                }
            )

        required = [
            CUSTOMER_ID,
            "Feature",
            "Feature Value",
            "SHAP Value",
            "Absolute SHAP"
        ]

        missing = [
            column
            for column in required
            if column not in data.columns
        ]

        if missing:

            raise ValueError(
                "Invalid local SHAP file. "
                f"Missing columns: {missing}"
            )

        data[
            CUSTOMER_ID
        ] = data[
            CUSTOMER_ID
        ].apply(
            self.normalize_customer_id
        )

        numeric_columns = [
            "Feature Value",
            "SHAP Value",
            "Absolute SHAP"
        ]

        for column in numeric_columns:

            data[
                column
            ] = pd.to_numeric(
                data[
                    column
                ],
                errors="coerce"
            )

        if "Impact Rank" in data.columns:

            data[
                "Impact Rank"
            ] = pd.to_numeric(
                data[
                    "Impact Rank"
                ],
                errors="coerce"
            )

        return (
            data
            .dropna(
                subset=[
                    "Feature",
                    "SHAP Value"
                ]
            )
            .reset_index(
                drop=True
            )
        )


    # =========================================================
    # ENSURE DATA LOADED
    # =========================================================

    def _ensure_churn_global(
        self
    ):

        if self.churn_global.empty:

            self.churn_global = (
                self._load_global(
                    self.churn_global_path
                )
            )


    def _ensure_churn_local(
        self
    ):

        if self.churn_local.empty:

            self.churn_local = (
                self._load_local(
                    self.churn_local_path
                )
            )


    def _ensure_clv_global(
        self
    ):

        if self.clv_global.empty:

            self.clv_global = (
                self._load_global(
                    self.clv_global_path
                )
            )


    def _ensure_clv_local(
        self
    ):

        if self.clv_local.empty:

            self.clv_local = (
                self._load_local(
                    self.clv_local_path
                )
            )


    # =========================================================
    # GLOBAL EXPLANATION
    # =========================================================

    @staticmethod
    def _format_global(
        data: pd.DataFrame,
        model_name: str
    ) -> dict:

        if data.empty:

            return {
                "model":
                    model_name,

                "available":
                    False,

                "feature_importance":
                    [],

                "interpretation_note":
                    (
                        "Global SHAP explanation "
                        "is not available."
                    )
            }

        features = []

        for _, row in data.iterrows():

            rank = row.get(
                "Importance Rank"
            )

            features.append(
                {
                    "feature":
                        str(
                            row[
                                "Feature"
                            ]
                        ),

                    "mean_absolute_shap":
                        float(
                            row[
                                "Mean Absolute SHAP"
                            ]
                        ),

                    "importance_rank":
                        (
                            int(
                                rank
                            )
                            if
                            pd.notna(
                                rank
                            )
                            else
                            None
                        )
                }
            )

        return {
            "model":
                model_name,

            "available":
                True,

            "feature_importance":
                features,

            "interpretation_note":
                (
                    "Mean absolute SHAP measures "
                    "average contribution magnitude "
                    "across explained customers. "
                    "It does not indicate causality."
                )
        }


    def global_churn(
        self
    ) -> dict:

        self._ensure_churn_global()

        return self._format_global(
            self.churn_global,
            "churn"
        )


    def global_clv(
        self
    ) -> dict:

        self._ensure_clv_global()

        return self._format_global(
            self.clv_global,
            "predicted_90_day_revenue"
        )


    # =========================================================
    # LOCAL CUSTOMER EXPLANATION
    # =========================================================

    @staticmethod
    def _format_contribution(
        row
    ) -> dict:

        shap_value = float(
            row[
                "SHAP Value"
            ]
        )

        if (
            "Impact Direction"
            in row.index
            and
            pd.notna(
                row.get(
                    "Impact Direction"
                )
            )
        ):

            direction = str(
                row[
                    "Impact Direction"
                ]
            )

        else:

            direction = (
                "Positive"
                if shap_value > 0
                else
                "Negative"
                if shap_value < 0
                else
                "Neutral"
            )

        rank = row.get(
            "Impact Rank"
        )

        return {

            "feature":
                str(
                    row[
                        "Feature"
                    ]
                ),

            "feature_value":
                float(
                    row[
                        "Feature Value"
                    ]
                ),

            "shap_value":
                shap_value,

            "absolute_shap":
                float(
                    row[
                        "Absolute SHAP"
                    ]
                ),

            "impact_direction":
                direction,

            "impact_rank":
                (
                    int(
                        rank
                    )
                    if
                    pd.notna(
                        rank
                    )
                    else
                    None
                )
        }


    def _customer_explanation(
        self,
        data: pd.DataFrame,
        customer_id,
        model_name: str,
        top_n: int = 3
    ) -> dict:

        normalized_id = (
            self.normalize_customer_id(
                customer_id
            )
        )

        if data.empty:

            return {
                "customer_id":
                    normalized_id,

                "model":
                    model_name,

                "explanation_available":
                    False,

                "top_positive_drivers":
                    [],

                "top_negative_drivers":
                    [],

                "all_contributions":
                    [],

                "interpretation_note":
                    (
                        "No precomputed SHAP "
                        "explanation is available."
                    )
            }

        customer_data = (
            data[
                data[
                    CUSTOMER_ID
                ]
                ==
                normalized_id
            ]
            .copy()
        )

        if customer_data.empty:

            return {
                "customer_id":
                    normalized_id,

                "model":
                    model_name,

                "explanation_available":
                    False,

                "top_positive_drivers":
                    [],

                "top_negative_drivers":
                    [],

                "all_contributions":
                    [],

                "interpretation_note":
                    (
                        "This customer was not included "
                        "in the precomputed SHAP sample."
                    )
            }

        customer_data = (
            customer_data
            .sort_values(
                by=
                    "Absolute SHAP",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )

        positive = (
            customer_data[
                customer_data[
                    "SHAP Value"
                ]
                >
                0
            ]
            .head(
                top_n
            )
        )

        negative = (
            customer_data[
                customer_data[
                    "SHAP Value"
                ]
                <
                0
            ]
            .head(
                top_n
            )
        )

        all_contributions = [
            self._format_contribution(
                row
            )
            for _, row
            in customer_data.iterrows()
        ]

        positive_contributions = [
            self._format_contribution(
                row
            )
            for _, row
            in positive.iterrows()
        ]

        negative_contributions = [
            self._format_contribution(
                row
            )
            for _, row
            in negative.iterrows()
        ]

        return {

            "customer_id":
                normalized_id,

            "model":
                model_name,

            "explanation_available":
                True,

            "top_positive_drivers":
                positive_contributions,

            "top_negative_drivers":
                negative_contributions,

            "all_contributions":
                all_contributions,

            "interpretation_note":
                (
                    "Positive SHAP contributions push "
                    "the model output upward relative "
                    "to its baseline; negative values "
                    "push it downward. SHAP explains "
                    "model behavior, not causality."
                )
        }


    # =========================================================
    # CHURN CUSTOMER
    # =========================================================

    def explain_churn_customer(
        self,
        customer_id,
        top_n: int = 3
    ) -> dict:

        self._ensure_churn_local()

        return self._customer_explanation(
            self.churn_local,
            customer_id,
            "churn",
            top_n
        )


    # =========================================================
    # CLV CUSTOMER
    # =========================================================

    def explain_clv_customer(
        self,
        customer_id,
        top_n: int = 3
    ) -> dict:

        self._ensure_clv_local()

        return self._customer_explanation(
            self.clv_local,
            customer_id,
            "predicted_90_day_revenue",
            top_n
        )


    # =========================================================
    # STATUS
    # =========================================================

    def status(
        self
    ) -> dict:

        churn_global_available = (
            self.file_available(
                self.churn_global_path
            )
        )

        churn_local_available = (
            self.file_available(
                self.churn_local_path
            )
        )

        clv_global_available = (
            self.file_available(
                self.clv_global_path
            )
        )

        clv_local_available = (
            self.file_available(
                self.clv_local_path
            )
        )

        churn_explained_customers = 0
        clv_explained_customers = 0

        if churn_local_available:

            try:

                self._ensure_churn_local()

                churn_explained_customers = int(
                    self.churn_local[
                        CUSTOMER_ID
                    ]
                    .nunique()
                )

            except Exception:

                churn_explained_customers = 0

        if clv_local_available:

            try:

                self._ensure_clv_local()

                clv_explained_customers = int(
                    self.clv_local[
                        CUSTOMER_ID
                    ]
                    .nunique()
                )

            except Exception:

                clv_explained_customers = 0

        return {

            "churn_global_available":
                churn_global_available,

            "churn_local_available":
                churn_local_available,

            "clv_global_available":
                clv_global_available,

            "clv_local_available":
                clv_local_available,

            "churn_explained_customers":
                churn_explained_customers,

            "clv_explained_customers":
                clv_explained_customers
        }