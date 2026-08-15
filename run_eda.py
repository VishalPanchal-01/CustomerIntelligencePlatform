from src.eda.exploratory_analysis import (
    ExploratoryAnalysis
)


eda = ExploratoryAnalysis()

df = eda.load_data()

df = eda.calculate_revenue(df)

total_revenue = eda.calculate_total_revenue(df)

print("\nDataset shape:")
print(df.shape)

print("\nTotal revenue:")
print(total_revenue)

print("\nOrder metrics:")
print(eda.calculate_order_metrics(df))

print("\nAverage order value:")
print(eda.calculate_average_order_value(df))

print("\nCustomer metrics:")
print(eda.calculate_customer_metrics(df))