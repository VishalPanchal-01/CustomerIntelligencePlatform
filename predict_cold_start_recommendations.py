from src.prediction.recommendation_predictor import (
    RecommendationPredictor
)


def main():

    predictor = (
        RecommendationPredictor()
    )

    unknown_customer_id = (
        "NEW_CUSTOMER_999999"
    )

    result = (
        predictor.recommend(
            customer_id=
                unknown_customer_id,

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
        "COLD START RECOMMENDATIONS"
    )

    print(
        "================================"
    )

    print(
        f"\nCustomer: "
        f"{result['customer_id']}"
    )

    print(
        f"Known Customer: "
        f"{result['known_customer']}"
    )

    print(
        f"Recommendation Source: "
        f"{result['recommendation_source']}"
    )

    for recommendation in (
        result[
            "recommendations"
        ]
    ):

        print(
            f"\n"
            f"{recommendation['Rank']}. "
            f"{recommendation['Description']}"
        )


if __name__ == "__main__":

    main()