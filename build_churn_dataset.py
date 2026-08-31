import os
from src.eda.exploratory_analysis import (ExploratoryAnalysis)
from src.preprocessing.data_cleaning import (DataCleaning)
from src.feature_engineering.churn_dataset import (ChurnDatasetBuilder)

def main():
    eda = ExploratoryAnalysis()
    cleaner = DataCleaning()
    builder = ChurnDatasetBuilder()

    df = eda.load_data()
    df = eda.calculate_revenue(df)

    cleaned_df = (cleaner.clean_for_churn(df))

    churn_df = (builder.build_dataset(cleaned_df,prediction_days=90))

    output_dir = os.path.join("artifacts","churn")
    os.makedirs(output_dir,exist_ok=True)
    output_path = os.path.join(output_dir,"customer_churn_dataset.csv")

    churn_df.to_csv(output_path,index=False)
    print("\nCustomer churn dataset created.")
    print(f"Shape: {churn_df.shape}")
    print("\nChurn distribution:")
    print(churn_df["Churn"].value_counts())
    print("\nChurn percentage:")

    print(churn_df["Churn"].value_counts(normalize=True)* 100)
    print(f"\nSaved to: {output_path}")

if __name__ == "__main__":
    main()