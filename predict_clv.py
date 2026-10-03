from src.prediction.clv_predictor import (
    CLVPredictor
)


def main():

    predictor = (
        CLVPredictor()
    )

    customer_data = {

        "Recency":
            20,

        "Frequency":
            8,

        "Monetary":
            4500,

        "TotalItems":
            350,

        "AverageOrderValue":
            562.50,

        "Tenure":
            300
    }

    prediction = (
        predictor.predict(
            customer_data
        )
    )

    print(
        "\n================================"
    )

    print(
        "CUSTOMER CLV PREDICTION"
    )

    print(
        "================================"
    )

    print(
        f"\nPredicted 90-Day Revenue: "
        f"{prediction['predicted_future_revenue']:.2f}"
    )

    print(
        f"CLV Value Band: "
        f"{prediction['value_band']}"
    )

    print(
        f"Prediction Horizon: "
        f"{prediction['prediction_horizon_days']} days"
    )

    print(
        f"Was Negative Before Clipping: "
        f"{prediction['was_negative_before_clipping']}"
    )


if __name__ == "__main__":

    main()