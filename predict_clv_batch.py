import os

import pandas as pd

from src.prediction.clv_predictor import (
    CLVPredictor
)


def main():

    output_directory = (
        "artifacts/clv/predictions"
    )

    output_path = os.path.join(
        output_directory,
        "customer_clv_predictions.csv"
    )

    customers = pd.DataFrame(
        [
            {
                "CustomerID":
                    101,

                "Recency":
                    10,

                "Frequency":
                    12,

                "Monetary":
                    6500,

                "TotalItems":
                    600,

                "AverageOrderValue":
                    541.67,

                "Tenure":
                    400
            },

            {
                "CustomerID":
                    102,

                "Recency":
                    45,

                "Frequency":
                    5,

                "Monetary":
                    1800,

                "TotalItems":
                    150,

                "AverageOrderValue":
                    360,

                "Tenure":
                    220
            },

            {
                "CustomerID":
                    103,

                "Recency":
                    120,

                "Frequency":
                    1,

                "Monetary":
                    150,

                "TotalItems":
                    10,

                "AverageOrderValue":
                    150,

                "Tenure":
                    80
            }
        ]
    )

    predictor = (
        CLVPredictor()
    )

    predictions = (
        predictor.predict_batch(
            customers
        )
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    predictions.to_csv(
        output_path,
        index=False
    )

    print(
        "\n================================"
    )

    print(
        "BATCH CLV PREDICTIONS"
    )

    print(
        "================================"
    )

    print(
        "\n"
        + predictions.to_string(
            index=False
        )
    )

    print(
        f"\nPredictions saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":

    main()