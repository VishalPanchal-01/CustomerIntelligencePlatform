from api.intelligence_service import (
    UnifiedIntelligenceService
)


# ============================================================
# FAKE CUSTOMER SERVICE
# ============================================================

class FakeCustomerService:

    @staticmethod
    def normalize_customer_id(
        value
    ):

        return str(
            int(
                float(
                    value
                )
            )
        )

    def get_customer(
        self,
        customer_id
    ):

        if str(
            customer_id
        ) == "999":

            return None

        return {

            "Customer ID":
                "101",

            "Customer Segment":
                "Segment 1",

            "Recency":
                100,

            "Frequency":
                5,

            "Monetary":
                2000,

            "TotalItems":
                150,

            "AverageOrderValue":
                400,

            "Tenure":
                300,

            "Churn Probability":
                0.82,

            "Churn Risk":
                "High",

            "Churn":
                1,

            "Predicted 90-Day Revenue":
                3500,

            "CLV Value Band":
                "High"
        }


# ============================================================
# FAKE RECOMMENDER
# ============================================================

class FakeRecommendationService:

    def recommend(
        self,
        customer_id,
        top_n
    ):

        return {

            "customer_id":
                str(
                    customer_id
                ),

            "recommendation_source":
                "Hybrid",

            "recommendation_mode":
                "next_purchase",

            "top_n":
                1,

            "cold_start":
                False,

            "recommendations": [
                {
                    "rank":
                        1,

                    "stock_code":
                        "A1",

                    "product":
                        "Product A",

                    "score":
                        0.9
                }
            ]
        }


# ============================================================
# FAKE EXPLAINABILITY
# ============================================================

class FakeExplainabilityService:

    def explain_churn_customer(
        self,
        customer_id,
        top_n
    ):

        return {

            "customer_id":
                str(
                    customer_id
                ),

            "model":
                "churn",

            "explanation_available":
                True,

            "top_positive_drivers":
                [],

            "top_negative_drivers":
                [],

            "all_contributions":
                [],

            "interpretation_note":
                "Test"
        }

    def explain_clv_customer(
        self,
        customer_id,
        top_n
    ):

        return {

            "customer_id":
                str(
                    customer_id
                ),

            "model":
                "predicted_90_day_revenue",

            "explanation_available":
                True,

            "top_positive_drivers":
                [],

            "top_negative_drivers":
                [],

            "all_contributions":
                [],

            "interpretation_note":
                "Test"
        }


# ============================================================
# CREATE SERVICE
# ============================================================

def create_service():

    return UnifiedIntelligenceService(
        customer_service=
            FakeCustomerService(),

        recommendation_service=
            FakeRecommendationService(),

        explainability_service=
            FakeExplainabilityService()
    )


# ============================================================
# CRITICAL PRIORITY
# ============================================================

def test_critical_business_priority():

    result = (
        UnifiedIntelligenceService
        .business_priority(
            churn_risk="High",
            clv_value_band="High"
        )
    )

    assert (
        result[
            "priority"
        ]
        ==
        "Critical"
    )


# ============================================================
# HIGH PRIORITY
# ============================================================

def test_high_business_priority():

    result = (
        UnifiedIntelligenceService
        .business_priority(
            churn_risk="Medium",
            clv_value_band="High"
        )
    )

    assert (
        result[
            "priority"
        ]
        ==
        "High"
    )


# ============================================================
# UNIFIED RESPONSE
# ============================================================

def test_unified_customer_intelligence():

    service = create_service()

    result = (
        service
        .get_intelligence(
            customer_id="101",
            recommendation_top_n=5,
            explanation_top_n=3
        )
    )

    assert (
        result[
            "customer_id"
        ]
        ==
        "101"
    )

    assert (
        result[
            "churn"
        ][
            "risk"
        ]
        ==
        "High"
    )

    assert (
        result[
            "clv"
        ][
            "value_band"
        ]
        ==
        "High"
    )

    assert (
        result[
            "business_decision"
        ][
            "priority"
        ]
        ==
        "Critical"
    )

    assert (
        result[
            "recommendation"
        ][
            "recommendations"
        ][0][
            "product"
        ]
        ==
        "Product A"
    )


# ============================================================
# CUSTOMER NOT FOUND
# ============================================================

def test_customer_not_found():

    service = create_service()

    result = (
        service
        .get_intelligence(
            customer_id="999"
        )
    )

    assert result is None