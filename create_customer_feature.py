from src.eda.exploratory_analysis import ExploratoryAnalysis
from src.preprocessing.data_cleaning import DataCleaning
from src.feature_engineering.customer_features import CustomerFeatureEngineering

eda = ExploratoryAnalysis()
cleaner = DataCleaning()
feature_engineering = (CustomerFeatureEngineering())

df = eda.load_data()
df = eda.calculate_revenue(df)
cleaned_df = (cleaner.clean_for_churn(df))
customer_features = (feature_engineering.create_customer_features(cleaned_df))

print("\nCustomer feature dataset:")
print(customer_features.head())
print("\nShape:")
print(customer_features.shape)
print("\nColumns:")
print(customer_features.columns.tolist())