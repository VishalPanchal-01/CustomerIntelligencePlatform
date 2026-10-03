import os
import pandas as pd

from src.preprocessing.data_cleaning import DataCleaning
from src.feature_engineering.clv_dataset import CLVDatasetBuilder


def main():

    # ============================================================
    # 1. CONFIGURATION
    # ============================================================

    input_path = os.path.join(
        "data",
        "processed",
        "retail.csv"
    )

    output_directory = os.path.join(
        "artifacts",
        "clv"
    )

    output_path = os.path.join(
        output_directory,
        "customer_clv_dataset.csv"
    )

    prediction_days = 90

    # ============================================================
    # 2. CHECK INPUT FILE
    # ============================================================

    if not os.path.exists(input_path):

        raise FileNotFoundError(
            "\nRetail dataset not found.\n\n"
            f"Expected file:\n"
            f"{os.path.abspath(input_path)}"
        )

    # ============================================================
    # 3. LOAD DATASET
    # ============================================================

    print(
        f"\nLoading dataset from:\n"
        f"{os.path.abspath(input_path)}"
    )

    df = pd.read_csv(
        input_path
    )

    print(
        "\nDataset loaded successfully."
    )

    print(
        f"Shape: {df.shape}"
    )

    print(
        "\nOriginal Columns:"
    )

    print(
        df.columns.tolist()
    )

    # ============================================================
    # 4. STANDARDIZE COLUMN NAMES
    # ============================================================

    column_mapping = {
        "Customer ID": "Customer ID",
        "Price": "Price"
    }

    df = df.rename(
        columns=column_mapping
    )

    print(
        "\nStandardized Columns:"
    )

    print(
        df.columns.tolist()
    )

    # ============================================================
    # 5. VALIDATE REQUIRED COLUMNS
    # ============================================================

    required_columns = [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    # ============================================================
    # 6. PREPARE DATE
    # ============================================================

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    # ============================================================
    # 7. CALCULATE REVENUE
    # ============================================================

    df["Revenue"] = (
        df["Quantity"]
        *
        df["Price"]
    )

    # ============================================================
    # 8. CLEAN VALID CUSTOMER PURCHASES
    # ============================================================

    cleaner = DataCleaning()

    cleaned_df = cleaner.clean_for_churn(
        df
    )

    print(
        "\nData cleaned successfully."
    )

    print(
        f"Cleaned Shape: "
        f"{cleaned_df.shape}"
    )

    # ============================================================
    # 9. BUILD CLV DATASET
    # ============================================================

    builder = CLVDatasetBuilder(
        prediction_days=prediction_days
    )

    clv_df = builder.build_dataset(
        cleaned_df
    )

    # ============================================================
    # 10. CREATE OUTPUT DIRECTORY
    # ============================================================

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # ============================================================
    # 11. SAVE CLV DATASET
    # ============================================================

    clv_df.to_csv(
        output_path,
        index=False
    )

    # ============================================================
    # 12. DISPLAY RESULTS
    # ============================================================

    print(
        "\n======================================"
    )

    print(
        "CUSTOMER LIFETIME VALUE DATASET"
    )

    print(
        "======================================"
    )

    print(
        f"\nPrediction Window: "
        f"{prediction_days} days"
    )

    print(
        f"\nDataset Shape: "
        f"{clv_df.shape}"
    )

    print(
        "\nColumns:"
    )

    for column in clv_df.columns:

        print(
            f"- {column}"
        )

    print(
        "\nFirst 5 Customers:"
    )

    print(
        clv_df.head()
    )

    print(
        "\nFuture Revenue Statistics:"
    )

    print(
        clv_df[
            "FutureRevenue"
        ].describe()
    )

    customers_with_revenue = (
        clv_df[
            "FutureRevenue"
        ]
        > 0
    ).sum()

    zero_revenue_customers = (
        clv_df[
            "FutureRevenue"
        ]
        == 0
    ).sum()

    print(
        "\nCustomers With Future Revenue:"
    )

    print(
        customers_with_revenue
    )

    print(
        "\nCustomers With Zero Future Revenue:"
    )

    print(
        zero_revenue_customers
    )

    print(
        f"\nCLV dataset saved to:"
        f"\n{output_path}"
    )


if __name__ == "__main__":

    main()