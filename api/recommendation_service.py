import os

import joblib
import numpy as np
import pandas as pd


CUSTOMER_ID = "Customer ID"


class RecommendationService:

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        model_path: str,
        batch_predictions_path: str
    ):

        self.model_path = model_path

        self.batch_predictions_path = (
            batch_predictions_path
        )

        self.model = None

        self.batch_predictions = (
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
    # SPLIT PIPE-SEPARATED VALUES
    # =========================================================

    @staticmethod
    def split_values(
        value
    ) -> list[str]:

        if pd.isna(
            value
        ):

            return []

        text = str(
            value
        ).strip()

        if (
            not text
            or
            text.lower()
            ==
            "nan"
        ):

            return []

        return [
            item.strip()
            for item in text.split("|")
            if item.strip()
        ]


    # =========================================================
    # MODEL AVAILABLE
    # =========================================================

    def model_available(
        self
    ) -> bool:

        return os.path.exists(
            self.model_path
        )


    # =========================================================
    # BATCH AVAILABLE
    # =========================================================

    def batch_predictions_available(
        self
    ) -> bool:

        return os.path.exists(
            self.batch_predictions_path
        )


    # =========================================================
    # LOAD MODEL
    # =========================================================

    def load_model(
        self
    ):

        if not self.model_available():

            raise FileNotFoundError(
                "Persisted recommendation model "
                "not found: "
                f"{self.model_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        return self.model


    # =========================================================
    # LOAD BATCH PREDICTIONS
    # =========================================================

    def load_batch_predictions(
        self
    ) -> pd.DataFrame:

        if not self.batch_predictions_available():

            raise FileNotFoundError(
                "Batch recommendation file "
                "not found: "
                f"{self.batch_predictions_path}"
            )

        data = pd.read_csv(
            self.batch_predictions_path
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

        if CUSTOMER_ID not in data.columns:

            raise ValueError(
                "Customer ID column not found "
                "in recommendation predictions."
            )

        data[
            CUSTOMER_ID
        ] = data[
            CUSTOMER_ID
        ].apply(
            self.normalize_customer_id
        )

        self.batch_predictions = (
            data
            .reset_index(
                drop=True
            )
        )

        return self.batch_predictions


    # =========================================================
    # ENSURE BATCH LOADED
    # =========================================================

    def _ensure_batch_loaded(
        self
    ) -> None:

        if self.batch_predictions.empty:

            self.load_batch_predictions()


    # =========================================================
    # SCORE COLUMN
    # =========================================================

    @staticmethod
    def _score_column(
        row
    ):

        possible_columns = [
            "Recommendation Scores",
            "Recommended Scores",
            "Scores"
        ]

        for column in possible_columns:

            if column in row.index:

                return column

        return None


    # =========================================================
    # BATCH RECOMMENDATIONS
    # =========================================================

    def _recommend_from_batch(
        self,
        customer_id,
        top_n: int
    ) -> dict | None:

        self._ensure_batch_loaded()

        normalized_id = (
            self.normalize_customer_id(
                customer_id
            )
        )

        match = (
            self.batch_predictions[
                self.batch_predictions[
                    CUSTOMER_ID
                ]
                ==
                normalized_id
            ]
        )

        if match.empty:

            return None

        row = match.iloc[0]

        products = self.split_values(
            row.get(
                "Recommended Products"
            )
        )

        stock_codes = self.split_values(
            row.get(
                "Recommended Stock Codes"
            )
        )

        score_column = (
            self._score_column(
                row
            )
        )

        if score_column is not None:

            raw_scores = self.split_values(
                row.get(
                    score_column
                )
            )

        else:

            raw_scores = []

        # -----------------------------------------------------
        # Support files that only persist rank-1 recommendation
        # -----------------------------------------------------

        if not products:

            top_product = row.get(
                "Top Recommended Product"
            )

            if pd.notna(
                top_product
            ):

                products = [
                    str(
                        top_product
                    )
                ]

        if not stock_codes:

            possible_code_columns = [
                "Top Recommended Stock Code",
                "Top Stock Code"
            ]

            for column in possible_code_columns:

                if (
                    column in row.index
                    and
                    pd.notna(
                        row.get(
                            column
                        )
                    )
                ):

                    stock_codes = [
                        str(
                            row.get(
                                column
                            )
                        )
                    ]

                    break

        scores = []

        for value in raw_scores:

            try:

                numeric = float(
                    value
                )

                if np.isfinite(
                    numeric
                ):

                    scores.append(
                        numeric
                    )

                else:

                    scores.append(
                        None
                    )

            except (
                TypeError,
                ValueError
            ):

                scores.append(
                    None
                )

        result_count = min(
            top_n,
            max(
                len(
                    products
                ),
                len(
                    stock_codes
                )
            )
        )

        recommendations = []

        for index in range(
            result_count
        ):

            recommendations.append(
                {
                    "rank":
                        index + 1,

                    "stock_code":
                        (
                            stock_codes[index]
                            if
                            index
                            <
                            len(
                                stock_codes
                            )
                            else
                            None
                        ),

                    "product":
                        (
                            products[index]
                            if
                            index
                            <
                            len(
                                products
                            )
                            else
                            None
                        ),

                    "score":
                        (
                            scores[index]
                            if
                            index
                            <
                            len(
                                scores
                            )
                            else
                            None
                        )
                }
            )

        source = row.get(
            "Recommendation Source",
            "Persisted Batch"
        )

        if pd.isna(
            source
        ):

            source = (
                "Persisted Batch"
            )

        return {

            "customer_id":
                normalized_id,

            "recommendation_source":
                str(
                    source
                ),

            "recommendation_mode":
                "next_purchase",

            "top_n":
                len(
                    recommendations
                ),

            "cold_start":
                False,

            "recommendations":
                recommendations
        }


    # =========================================================
    # MODEL METHOD DETECTION
    # =========================================================

    def _model_recommend(
        self,
        customer_id,
        top_n: int
    ):

        if self.model is None:

            if not self.model_available():

                return None

            self.load_model()

        normalized_id = (
            self.normalize_customer_id(
                customer_id
            )
        )

        # -----------------------------------------------------
        # We only call interfaces that actually exist.
        # -----------------------------------------------------

        possible_methods = [
            "recommend",
            "recommend_products",
            "get_recommendations"
        ]

        for method_name in possible_methods:

            if hasattr(
                self.model,
                method_name
            ):

                method = getattr(
                    self.model,
                    method_name
                )

                try:

                    result = method(
                        normalized_id,
                        top_n=top_n
                    )

                except TypeError:

                    try:

                        result = method(
                            normalized_id,
                            top_n
                        )

                    except TypeError:

                        continue

                return result

        return None


    # =========================================================
    # COLD START SUPPORT DETECTION
    # =========================================================

    def cold_start_supported(
        self
    ) -> bool:

        if self.model is None:

            if not self.model_available():

                return False

            try:

                self.load_model()

            except Exception:

                return False

        # -----------------------------------------------------
        # Explicit signal
        # -----------------------------------------------------

        if hasattr(
            self.model,
            "supports_cold_start"
        ):

            try:

                return bool(
                    self.model.supports_cold_start
                )

            except Exception:

                pass

        # -----------------------------------------------------
        # Known popularity fallback.
        # This is only reported when such an object exists.
        # -----------------------------------------------------

        if hasattr(
            self.model,
            "popularity_model"
        ):

            return True

        return False


    # =========================================================
    # PUBLIC RECOMMEND
    # =========================================================

    def recommend(
        self,
        customer_id,
        top_n: int = 5
    ) -> dict:

        if top_n < 1:

            raise ValueError(
                "top_n must be at least 1."
            )

        if top_n > 50:

            raise ValueError(
                "top_n cannot exceed 50."
            )

        # -----------------------------------------------------
        # Reliable production lookup first
        # -----------------------------------------------------

        try:

            batch_result = (
                self._recommend_from_batch(
                    customer_id,
                    top_n
                )
            )

        except FileNotFoundError:

            batch_result = None

        if (
            batch_result is not None
            and
            batch_result[
                "recommendations"
            ]
        ):

            return batch_result

        # -----------------------------------------------------
        # Customer not present in persisted batch.
        #
        # Try model only if it exposes a usable recommendation
        # interface. We do not invent a cold-start fallback.
        # -----------------------------------------------------

        model_result = (
            self._model_recommend(
                customer_id,
                top_n
            )
        )

        if model_result is None:

            raise LookupError(
                "No recommendation is available "
                f"for customer {customer_id}. "
                "The selected persisted recommender "
                "does not expose a supported cold-start "
                "recommendation interface for this customer."
            )

        # -----------------------------------------------------
        # Normalize common model return formats
        # -----------------------------------------------------

        recommendations = (
            self._normalize_model_result(
                model_result,
                top_n
            )
        )

        if not recommendations:

            raise LookupError(
                "The recommender returned no products "
                f"for customer {customer_id}."
            )

        return {

            "customer_id":
                self.normalize_customer_id(
                    customer_id
                ),

            "recommendation_source":
                type(
                    self.model
                ).__name__,

            "recommendation_mode":
                "next_purchase",

            "top_n":
                len(
                    recommendations
                ),

            "cold_start":
                True,

            "recommendations":
                recommendations
        }


    # =========================================================
    # NORMALIZE MODEL RESULT
    # =========================================================

    @staticmethod
    def _normalize_model_result(
        result,
        top_n: int
    ) -> list[dict]:

        recommendations = []

        # -----------------------------------------------------
        # DataFrame
        # -----------------------------------------------------

        if isinstance(
            result,
            pd.DataFrame
        ):

            data = result.head(
                top_n
            )

            for position, (
                _,
                row
            ) in enumerate(
                data.iterrows()
            ):

                product = (
                    row.get(
                        "Product"
                    )
                    or
                    row.get(
                        "Description"
                    )
                )

                code = (
                    row.get(
                        "Stock Code"
                    )
                    or
                    row.get(
                        "StockCode"
                    )
                )

                score = row.get(
                    "Score"
                )

                if pd.isna(
                    score
                ):

                    score = None

                recommendations.append(
                    {
                        "rank":
                            position + 1,

                        "stock_code":
                            (
                                str(
                                    code
                                )
                                if
                                code is not None
                                and
                                not pd.isna(
                                    code
                                )
                                else
                                None
                            ),

                        "product":
                            (
                                str(
                                    product
                                )
                                if
                                product is not None
                                and
                                not pd.isna(
                                    product
                                )
                                else
                                None
                            ),

                        "score":
                            (
                                float(
                                    score
                                )
                                if
                                score is not None
                                else
                                None
                            )
                    }
                )

            return recommendations

        # -----------------------------------------------------
        # list / tuple
        # -----------------------------------------------------

        if isinstance(
            result,
            (
                list,
                tuple
            )
        ):

            for position, item in enumerate(
                result[
                    :top_n
                ]
            ):

                if isinstance(
                    item,
                    dict
                ):

                    product = (
                        item.get(
                            "product"
                        )
                        or
                        item.get(
                            "Product"
                        )
                        or
                        item.get(
                            "description"
                        )
                    )

                    code = (
                        item.get(
                            "stock_code"
                        )
                        or
                        item.get(
                            "Stock Code"
                        )
                        or
                        item.get(
                            "StockCode"
                        )
                    )

                    score = (
                        item.get(
                            "score"
                        )
                        or
                        item.get(
                            "Score"
                        )
                    )

                else:

                    product = str(
                        item
                    )

                    code = None

                    score = None

                recommendations.append(
                    {
                        "rank":
                            position + 1,

                        "stock_code":
                            (
                                str(
                                    code
                                )
                                if
                                code is not None
                                else
                                None
                            ),

                        "product":
                            (
                                str(
                                    product
                                )
                                if
                                product is not None
                                else
                                None
                            ),

                        "score":
                            (
                                float(
                                    score
                                )
                                if
                                score is not None
                                else
                                None
                            )
                    }
                )

        return recommendations


    # =========================================================
    # MODEL STATUS
    # =========================================================

    def model_status(
        self
    ) -> dict:

        known_customers = 0

        if self.batch_predictions_available():

            try:

                self._ensure_batch_loaded()

                known_customers = int(
                    self.batch_predictions[
                        CUSTOMER_ID
                    ]
                    .nunique()
                )

            except Exception:

                known_customers = 0

        return {

            "model_available":
                self.model_available(),

            "model_loaded":
                self.model is not None,

            "batch_predictions_available":
                self.batch_predictions_available(),

            "batch_predictions_loaded":
                not self.batch_predictions.empty,

            "model_path":
                self.model_path,

            "batch_predictions_path":
                self.batch_predictions_path,

            "known_customers":
                known_customers,

            "cold_start_supported":
                self.cold_start_supported()
        }