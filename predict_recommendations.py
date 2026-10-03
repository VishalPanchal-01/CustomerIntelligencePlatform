from src.prediction.recommendation_predictor import (
    RecommendationPredictor
)


def main():

    predictor = (
        RecommendationPredictor()
    )

    # ---------------------------------
    # Replace this with a valid
    # CustomerID from your dataset.
    # ---------------------------------

    customer_id = 12347

    # =================================
    # NEXT PURCHASE MODE
    # =================================

    result = (
        predictor.recommend(
            customer_id=
                customer_id,

            top_k=
                10,

            mode=
                "next_purchase"
        )
    )

    print(
        "\n================================"
    )

    print(
        "NEXT PURCHASE RECOMMENDATIONS"
    )

    print(
        "================================"
    )

    print(
        f"\nCustomer ID: "
        f"{result['customer_id']}"
    )

    print(
        f"Known Customer: "
        f"{result['known_customer']}"
    )

    print(
        f"Source: "
        f"{result['recommendation_source']}"
    )

    print(
        f"Recommendations: "
        f"{result['recommendation_count']}"
    )

    for recommendation in (
        result[
            "recommendations"
        ]
    ):

        print(
            f"\n"
            f"{recommendation['Rank']}. "
            f"{recommendation['StockCode']} | "
            f"{recommendation['Description']} | "
            f"Score="
            f"{recommendation['Score']:.4f}"
        )


if __name__ == "__main__":

    main()