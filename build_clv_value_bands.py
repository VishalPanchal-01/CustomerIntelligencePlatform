import json
import os

import pandas as pd

from src.preprocessing.clv_preprocessing import (
    CLVPreprocessor
)

from src.training.clv_data_split import (
    CLVDataSplitter
)


def main():

    # ---------------------------------
    # Paths
    # ---------------------------------

    input_path = (
        "artifacts/clv/"
        "customer_clv_dataset.csv"
    )

    output_path = (
        "models/clv/"
        "clv_value_bands.json"
    )

    # ---------------------------------
    # Load dataset
    # ---------------------------------

    df = pd.read_csv(
        input_path
    )

    # ---------------------------------
    # Prepare
    # ---------------------------------

    preprocessor = (
        CLVPreprocessor()
    )

    (
        X,
        y_raw,
        y_log
    ) = preprocessor.prepare_features(
        df
    )

    # ---------------------------------
    # Same final split
    # ---------------------------------

    splitter = (
        CLVDataSplitter()
    )

    (
        X_train,
        X_test,
        y_raw_train,
        y_raw_test,
        y_log_train,
        y_log_test
    ) = splitter.split_data(
        X,
        y_raw,
        y_log,
        test_size=0.20,
        random_state=42
    )

    # ---------------------------------
    # Calculate thresholds using
    # TRAINING TARGET ONLY
    # ---------------------------------

    low_upper_bound = float(
        y_raw_train.quantile(
            0.33
        )
    )

    medium_upper_bound = float(
        y_raw_train.quantile(
            0.66
        )
    )

    bands = {

        "method":
            "Training target quantiles",

        "prediction_horizon_days":
            90,

        "low_upper_bound":
            low_upper_bound,

        "medium_upper_bound":
            medium_upper_bound,

        "definitions": {

            "Low":
                (
                    f"Predicted revenue <= "
                    f"{low_upper_bound:.2f}"
                ),

            "Medium":
                (
                    f"Predicted revenue > "
                    f"{low_upper_bound:.2f} "
                    f"and <= "
                    f"{medium_upper_bound:.2f}"
                ),

            "High":
                (
                    f"Predicted revenue > "
                    f"{medium_upper_bound:.2f}"
                )
        }
    }

    # ---------------------------------
    # Save
    # ---------------------------------

    os.makedirs(
        os.path.dirname(
            output_path
        ),
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            bands,
            file,
            indent=4
        )

    # ---------------------------------
    # Output
    # ---------------------------------

    print(
        "\n================================"
    )

    print(
        "CLV VALUE BANDS"
    )

    print(
        "================================"
    )

    print(
        f"\nLow:"
        f"\n0 - {low_upper_bound:.2f}"
    )

    print(
        f"\nMedium:"
        f"\n{low_upper_bound:.2f} "
        f"- {medium_upper_bound:.2f}"
    )

    print(
        f"\nHigh:"
        f"\n> {medium_upper_bound:.2f}"
    )

    print(
        f"\nConfiguration saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":

    main()