class UnifiedIntelligenceService:

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        customer_service,
        recommendation_service,
        explainability_service
    ):

        self.customer_service = (
            customer_service
        )

        self.recommendation_service = (
            recommendation_service
        )

        self.explainability_service = (
            explainability_service
        )


    # =========================================================
    # BUSINESS PRIORITY
    # =========================================================

    @staticmethod
    def business_priority(
        churn_risk,
        clv_value_band
    ) -> dict:

        churn = (
            str(
                churn_risk
            )
            .strip()
            .lower()
            if churn_risk is not None
            else ""
        )

        value = (
            str(
                clv_value_band
            )
            .strip()
            .lower()
            if clv_value_band is not None
            else ""
        )

        # -----------------------------------------------------
        # Same logic used in dashboard:
        #
        # Critical = High churn + High value
        # High = Medium churn + High value
        #        OR High churn + Medium value
        # Medium = Low churn + High value
        #          OR Medium churn + Medium value
        #          OR High churn + Low value
        # Low = everything else
        # -----------------------------------------------------

        if (
            churn == "high"
            and
            value == "high"
        ):

            return {
                "priority":
                    "Critical",

                "recommended_action":
                    (
                        "Prioritize immediate retention. "
                        "Use personalized outreach, "
                        "service recovery or a targeted "
                        "retention offer while protecting "
                        "customer value."
                    )
            }

        if (
            (
                churn == "medium"
                and
                value == "high"
            )
            or
            (
                churn == "high"
                and
                value == "medium"
            )
        ):

            return {
                "priority":
                    "High",

                "recommended_action":
                    (
                        "Actively monitor and engage this "
                        "customer with relevant personalized "
                        "offers and retention messaging."
                    )
            }

        if (
            (
                churn == "low"
                and
                value == "high"
            )
            or
            (
                churn == "medium"
                and
                value == "medium"
            )
            or
            (
                churn == "high"
                and
                value == "low"
            )
        ):

            return {
                "priority":
                    "Medium",

                "recommended_action":
                    (
                        "Maintain engagement and use "
                        "appropriate cross-sell, retention "
                        "or nurture actions based on the "
                        "customer profile."
                    )
            }

        return {
            "priority":
                "Low",

            "recommended_action":
                (
                    "Continue regular engagement and "
                    "monitor for meaningful changes in "
                    "customer risk or value."
                )
        }


    # =========================================================
    # CHURN SECTION
    # =========================================================

    @staticmethod
    def build_churn_section(
        customer: dict
    ) -> dict:

        return {

            "probability":
                customer.get(
                    "Churn Probability"
                ),

            "risk":
                customer.get(
                    "Churn Risk"
                ),

            "predicted_class":
                customer.get(
                    "Churn"
                )
        }


    # =========================================================
    # CLV SECTION
    # =========================================================

    @staticmethod
    def build_clv_section(
        customer: dict
    ) -> dict:

        return {

            "predicted_90_day_revenue":
                customer.get(
                    "Predicted 90-Day Revenue"
                ),

            "value_band":
                customer.get(
                    "CLV Value Band"
                )
        }


    # =========================================================
    # CUSTOMER PROFILE
    # =========================================================

    @staticmethod
    def build_profile(
        customer: dict
    ) -> dict:

        profile_columns = [
            "Customer ID",
            "Customer Segment",
            "Recency",
            "Frequency",
            "Monetary",
            "TotalItems",
            "AverageOrderValue",
            "Tenure"
        ]

        return {
            column:
                customer.get(
                    column
                )
            for column in profile_columns
            if column in customer
        }


    # =========================================================
    # UNIFIED INTELLIGENCE
    # =========================================================

    def get_intelligence(
        self,
        customer_id,
        recommendation_top_n: int = 5,
        explanation_top_n: int = 3
    ) -> dict | None:

        customer = (
            self.customer_service
            .get_customer(
                customer_id
            )
        )

        if customer is None:

            return None

        normalized_id = (
            self.customer_service
            .normalize_customer_id(
                customer_id
            )
        )

        # -----------------------------------------------------
        # Recommendation
        # -----------------------------------------------------

        try:

            recommendation = (
                self.recommendation_service
                .recommend(
                    customer_id=
                        normalized_id,

                    top_n=
                        recommendation_top_n
                )
            )

        except (
            LookupError,
            FileNotFoundError,
            ValueError
        ):

            recommendation = None

        # -----------------------------------------------------
        # Churn explanation
        # -----------------------------------------------------

        churn_explanation = (
            self.explainability_service
            .explain_churn_customer(
                customer_id=
                    normalized_id,

                top_n=
                    explanation_top_n
            )
        )

        # -----------------------------------------------------
        # CLV explanation
        # -----------------------------------------------------

        clv_explanation = (
            self.explainability_service
            .explain_clv_customer(
                customer_id=
                    normalized_id,

                top_n=
                    explanation_top_n
            )
        )

        # -----------------------------------------------------
        # Business decision
        # -----------------------------------------------------

        business_decision = (
            self.business_priority(
                churn_risk=
                    customer.get(
                        "Churn Risk"
                    ),

                clv_value_band=
                    customer.get(
                        "CLV Value Band"
                    )
            )
        )

        return {

            "customer_id":
                normalized_id,

            "profile":
                self.build_profile(
                    customer
                ),

            "churn":
                self.build_churn_section(
                    customer
                ),

            "clv":
                self.build_clv_section(
                    customer
                ),

            "recommendation":
                recommendation,

            "churn_explanation":
                churn_explanation,

            "clv_explanation":
                clv_explanation,

            "business_decision":
                business_decision
        }