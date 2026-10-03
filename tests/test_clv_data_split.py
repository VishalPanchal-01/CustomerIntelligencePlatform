import numpy as np
import pandas as pd

from src.training.clv_data_split import (
    CLVDataSplitter
)


def test_clv_data_split():

    X = pd.DataFrame(
        {
            "Recency":
                list(
                    range(
                        1,
                        21
                    )
                ),

            "Frequency":
                list(
                    range(
                        20,
                        0,
                        -1
                    )
                ),

            "Monetary":
                [
                    value * 100
                    for value
                    in range(
                        1,
                        21
                    )
                ],

            "TotalItems":
                [
                    value * 10
                    for value
                    in range(
                        1,
                        21
                    )
                ],

            "AverageOrderValue":
                [
                    100
                    for _ in range(
                        20
                    )
                ],

            "Tenure":
                [
                    value * 20
                    for value
                    in range(
                        1,
                        21
                    )
                ]
        }
    )

    y_raw = pd.Series(
        [
            value * 50
            for value
            in range(
                20
            )
        ],
        name="FutureRevenue"
    )

    y_log = pd.Series(
        np.log1p(
            y_raw
        ),
        name="LogFutureRevenue"
    )

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

    assert (
        len(X_train)
        == 16
    )

    assert (
        len(X_test)
        == 4
    )

    assert (
        len(y_raw_train)
        == 16
    )

    assert (
        len(y_raw_test)
        == 4
    )

    assert (
        len(y_log_train)
        == 16
    )

    assert (
        len(y_log_test)
        == 4
    )

    assert (
        X_train.index.tolist()
        ==
        y_raw_train.index.tolist()
    )

    assert (
        X_train.index.tolist()
        ==
        y_log_train.index.tolist()
    )

    assert (
        X_test.index.tolist()
        ==
        y_raw_test.index.tolist()
    )

    assert (
        X_test.index.tolist()
        ==
        y_log_test.index.tolist()
    )

    assert set(
        X_train.index
    ).isdisjoint(
        set(
            X_test.index
        )
    )