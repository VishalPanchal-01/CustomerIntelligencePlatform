import pandas as pd

from src.feature_engineering.churn_target import ChurnTargetCreation



def test_create_churn_target():
    df = pd.DataFrame({
        "Invoice": ["10001","10002","10003","10004","10005"],
        "InvoiceDate": ["2021-01-01","2021-01-10","2021-02-01","2021-04-01","2021-04-15"],
        "Customer ID": [101,102,101,102,103],
        "Quantity": [2,3,1,2,1],
        "Price": [10,20,10,20,30],
        "Revenue": [20,60,10,40,30]
    })

    target_creator = (ChurnTargetCreation())
    result = (target_creator.create_churn_target(df,prediction_days=90))

    assert result is not None
    assert "Customer ID" in result.columns
    assert "Churn" in result.columns
    assert set(result["Churn"].unique()).issubset({0,1})