import pandas as pd

from src.feature_engineering.customer_features import CustomerFeatureEngineering



def test_customer_feature_engineering():
    df = pd.DataFrame({
        "Invoice": ["10001","10001","10002","10003"],
        "InvoiceDate": ["2011-01-01","2011-01-01","2011-01-10","2011-02-01"],
        "Customer ID": [101,101,101,102],
        "Quantity": [2,3,1,4],
        "Price": [10,20,50,25],
        "Revenue": [20,60,50,100]
    })

    feature_engineering = (CustomerFeatureEngineering())
    result = (feature_engineering.create_customer_features(df))

    assert result is not None
    assert len(result) == 2

    expected_columns = ["Customer ID","Recency","Frequency","Monetary","TotalItems","FirstPurchaseDate","LastPurchaseDate","AverageOrderValue","Tenure"]

    for column in expected_columns:
        assert column in result.columns

    customer_101 = result[result["Customer ID"] == 101].iloc[0]

    assert customer_101["Frequency"] == 2
    assert customer_101["Monetary"] == 130
    assert customer_101["TotalItems"] == 6
    assert (customer_101["AverageOrderValue"]== 65)

    customer_102 = result[result["Customer ID"] == 102].iloc[0]

    assert customer_102["Frequency"] == 1
    assert customer_102["Monetary"] == 100
    assert customer_102["TotalItems"] == 4