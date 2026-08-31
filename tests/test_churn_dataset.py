import pandas as pd
from src.feature_engineering.churn_dataset import ChurnDatasetBuilder

def test_build_churn_dataset():
    df = pd.DataFrame({
        "Invoice": ["10001","10002","10003","10004","10005","10006"],
        "InvoiceDate": ["2021-01-01","2021-01-10","2021-02-01","2021-03-01","2021-04-01","2021-04-15"],
        "Customer ID": [101,101,102,102,103,103],
        "Quantity": [2,3,1,2,5,1],
        "Price": [10,20,30,10,5,5],
        "Revenue": [20,60,30,20,25,5]
    })

    builder = ChurnDatasetBuilder()
    result = builder.build_dataset(df,prediction_days=90)

    assert result is not None
    assert "Customer ID" in result.columns
    assert "Churn" in result.columns
    assert "Recency" in result.columns
    assert "Frequency" in result.columns
    assert "Monetary" in result.columns
    assert "AverageOrderValue" in result.columns
    assert "Tenure" in result.columns
    assert result["Customer ID"].is_unique
    assert set(result["Churn"].unique()).issubset({0, 1})