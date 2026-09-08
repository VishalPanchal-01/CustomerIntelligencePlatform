import pandas as pd

from src.evaluation.cross_validation import ChurnCrossValidation

def test_churn_cross_validation():
    X = pd.DataFrame({
            "Recency": [
                10, 20, 30, 40, 50,
                60, 70, 80, 90, 100,
                110, 120, 130, 140, 150,
                160, 170, 180, 190, 200
                ],
            "Frequency": [
                10, 9, 8, 7, 6,
                5, 4, 3, 3, 2,
                2, 2, 1, 1, 1,
                1, 1, 1, 1, 1
            ],
            "Monetary": [
                2000, 1800, 1600, 1500, 1400,
                1200, 1000, 900, 800, 700,
                600, 500, 400, 350, 300,
                250, 200, 150, 100, 50
            ],
            "TotalItems": [
                200, 180, 160, 150, 140,
                120, 100, 90, 80, 70,
                60, 50, 40, 35, 30,
                25, 20, 15, 10, 5
            ],
            "AverageOrderValue": [
                200, 200, 200, 214, 233,
                240, 250, 300, 267, 350,
                300, 250, 400, 350, 300,
                250, 200, 150, 100, 50
            ],
            "Tenure": [
                400, 390, 380, 370, 360,
                350, 340, 330, 320, 310,
                200, 180, 160, 140, 120,
                100, 80, 60, 40, 20
            ]
        })

    y = pd.Series(
        [
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1
        ]
    )

    validator = (ChurnCrossValidation())

    result = (validator.evaluate_models(X,y,n_splits=5))

    assert result is not None

    assert len(result) == 2

    assert "Model" in result.columns

    assert "F1Mean" in result.columns

    assert "F1Std" in result.columns

    assert "ROCAUCMean" in result.columns

    assert "ROCAUCStd" in result.columns

    assert set(result["Model"]) == {"Logistic Regression","Random Forest"}