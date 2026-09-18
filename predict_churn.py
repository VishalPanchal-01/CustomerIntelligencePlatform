from src.prediction.churn_predictor import (
    ChurnPredictor
)


def main():

    predictor = ChurnPredictor()

    customer_data = {
        "Recency": 120,
        "Frequency": 2,
        "Monetary": 450.0,
        "TotalItems": 25,
        "AverageOrderValue": 225.0,
        "Tenure": 90
    }

    result = predictor.predict(
        customer_data
    )

    print(
        "\nCustomer Churn Prediction:"
    )

    print(
        f"Prediction: "
        f"{result['prediction']}"
    )

    if (
        result["churn_probability"]
        is not None
    ):

        print(
            f"Churn Probability: "
            f"{result['churn_probability']:.4f}"
        )

    else:

        print(
            "Churn Probability: "
            "Not available"
        )

    print(
        f"Risk Level: "
        f"{result['risk_level']}"
    )


if __name__ == "__main__":

    main()