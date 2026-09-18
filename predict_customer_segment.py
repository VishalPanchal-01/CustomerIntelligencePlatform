from src.prediction.segment_predictor import (
    SegmentPredictor
)


def main():

    predictor = (
        SegmentPredictor()
    )

    customer_data = {
        "Recency": 20,
        "Frequency": 8,
        "Monetary": 4500.0,
        "TotalItems": 350,
        "AverageOrderValue": 562.5,
        "Tenure": 300
    }

    result = (
        predictor.predict_segment(
            customer_data
        )
    )

    print(
        "\n================================"
    )

    print(
        "CUSTOMER SEGMENT PREDICTION"
    )

    print(
        "================================"
    )

    print(
        f"\nCluster: "
        f"{result['cluster']}"
    )

    print(
        f"\nSegment: "
        f"{result['segment_name']}"
    )

    print(
        "\nRecommendation:"
    )

    print(
        result[
            "recommendation"
        ]
    )


if __name__ == "__main__":

    main()