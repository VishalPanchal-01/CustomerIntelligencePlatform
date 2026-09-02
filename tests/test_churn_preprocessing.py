import pandas as pd

from src.preprocessing.churn_preprocessing import ChurnPreprocessor

def test_prepare_features():
    df = pd.DataFrame({
            "Customer ID": [101,102,103],
            "Recency": [10,50,120],
            "Frequency": [5,3,1],
            "Monetary": [1000,500,100],
            "TotalItems": [100,50,10],
            "AverageOrderValue": [200,166.67,100],
            "Tenure": [300,150,20],
            "Churn": [0,0,1]
        })

    preprocessor = (ChurnPreprocessor())

    X, y = (preprocessor.prepare_features(df))

    assert X is not None

    assert y is not None

    assert X.shape == (3,6)

    assert len(y) == 3

    assert "Customer ID" not in X.columns

    assert "Churn" not in X.columns

    assert list(X.columns) == ["Recency","Frequency","Monetary","TotalItems","AverageOrderValue","Tenure"]

    assert set(y.unique()).issubset({0, 1})