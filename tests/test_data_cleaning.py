import pandas as pd

from src.preprocessing.data_cleaning import (DataCleaning)

def test_clean_for_churn():
    df = pd.DataFrame({
        "Invoice": ["10001","10002","C10003","10004"],
        "StockCode": ["A","B","C","D"],
        "Quantity": [5,-2,-1,3],
        "InvoiceDate": ["2011-01-01","2011-01-02","2011-01-03","2011-01-04"],
        "Price": [10,10,10,5],
        "Customer ID": [101,102,103,None],
        "Country": ["UK","UK","UK","UK"],
        "Revenue": [50,-20,-10,15]
    })

    cleaner = DataCleaning()
    cleaned_df = (cleaner.clean_for_churn(df))

    # Only the first row should remain
    # because it is the only valid
    # identifiable purchase.

    assert len(cleaned_df) == 1
    assert (cleaned_df.iloc[0]["Customer ID"]== 101)

    assert (cleaned_df.iloc[0]["Revenue"]== 50)

    assert (cleaned_df["Quantity"] > 0).all()

    assert (cleaned_df["Price"] > 0).all()