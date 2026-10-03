import numpy as np
import pandas as pd

from src.evaluation.clv_cross_validation import (
    CLVCrossValidator
)


def test_clv_cross_validation():

    X = pd.DataFrame(
        {
            "Recency": [
                10,
                15,
                20,
                25,
                30,
                35,
                40,
                45,
                50,
                55,
                60,
                65,
                70,
                75,
                80,
                85,
                90,
                95,
                100,
                105
            ],

            "Frequency": [
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
                9,
                8,
                7,
                6,
                5,
                4,
                3,
                2,
                1
            ],

            "Monetary": [
                10000,
                9500,
                9000,
                8500,
                8000,
                7500,
                7000,
                6500,
                6000,
                5500,
                5000,
                4500,
                4000,
                3500,
                3000,
                2500,
                2000,
                1500,
                1000,
                500
            ],

            "TotalItems": [
                1000,
                950,
                900,
                850,
                800,
                750,
                700,
                650,
                600,
                550,
                500,
                450,
                400,
                350,
                300,
                250,
                200,
                150,
                100,
                50
            ],

            "AverageOrderValue": [
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500,
                500
            ],

            "Tenure": [
                500,
                485,
                470,
                455,
                440,
                425,
                410,
                395,
                380,
                365,
                350,
                335,
                320,
                305,
                290,
                275,
                260,
                245,
                230,
                215
            ]
        }
    )

    y_raw = pd.Series(
        [
            5000,
            4750,
            4500,
            4250,
            4000,
            3750,
            3500,
            3250,
            3000,
            2750,
            2500,
            2250,
            2000,
            1750,
            1500,
            1250,
            1000,
            750,
            500,
            250
        ],
        name="FutureRevenue"
    )

    y_log = pd.Series(
        np.log1p(
            y_raw
        ),
        name="LogFutureRevenue"
    )

    validator = (
        CLVCrossValidator(
            n_splits=4,
            random_state=42
        )
    )

    result = (
        validator.evaluate_models(
            X,
            y_raw,
            y_log
        )
    )

    assert result is not None

    assert (
        len(result)
        == 6
    )

    expected_columns = [
        "Model",
        "Target",
        "MAEMean",
        "MAEStd",
        "RMSEMean",
        "RMSEStd",
        "R2Mean",
        "R2Std",
        "TotalNegativePredictions"
    ]

    assert (
        result.columns.tolist()
        ==
        expected_columns
    )

    assert set(
        result[
            "Model"
        ]
    ) == {
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    }

    assert set(
        result[
            "Target"
        ]
    ) == {
        "Raw",
        "Log"
    }

    assert (
        result[
            "MAEMean"
        ]
        >= 0
    ).all()

    assert (
        result[
            "MAEStd"
        ]
        >= 0
    ).all()

    assert (
        result[
            "RMSEMean"
        ]
        >= 0
    ).all()

    assert (
        result[
            "RMSEStd"
        ]
        >= 0
    ).all()

    assert np.isfinite(
        result[
            "R2Mean"
        ]
    ).all()

    assert np.isfinite(
        result[
            "R2Std"
        ]
    ).all()