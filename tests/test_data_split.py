import pandas as pd

from src.training.data_split import ChurnDataSplitter

def test_split_data():
    X = pd.DataFrame({
            "Recency": range(100),
            "Frequency": range(100),
            "Monetary": range(100),
            "TotalItems": range(100),
            "AverageOrderValue": range(100),
            "Tenure": range(100)
        })

    y = pd.Series([0] * 70 + [1] * 30)

    splitter = (ChurnDataSplitter())

    X_train,X_test,y_train,y_test = splitter.split_data(X,y,test_size=0.2,random_state=42)

    assert len(X_train) == 80

    assert len(X_test) == 20

    assert len(y_train) == 80

    assert len(y_test) == 20

    assert (len(X_train.columns)== 6)

    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})