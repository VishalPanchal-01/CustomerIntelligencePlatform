from src.eda.exploratory_analysis import (
    ExploratoryAnalysis
)


eda = ExploratoryAnalysis()

df = eda.load_data()

df = eda.calculate_revenue(df)

total_revenue = eda.calculate_total_revenue(df)

df = eda.prepare_date_column(df)

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



print("\nMonthly revenue:")
print(eda.calculate_monthly_revenue(df))

print("\nMonthly orders:")
print(eda.calculate_monthly_orders(df))

print("\nMonthly customers:")
print(eda.calculate_monthly_customers(df))


print("\nCountry metrics:")
country_metrics = (eda.calculate_country_metrics(df))

print("\nCustomers per country:")
print(country_metrics["customers_per_country"])

print("\nOrders per country:")
print(country_metrics["orders_per_country"])

print("\nRevenue per country:")
print(country_metrics["revenue_per_country"])


print("\nRevenue share by country:")
print(eda.calculate_country_revenue_share(df).head(10))


print("\nProduct metrics:")
product_metrics = (eda.calculate_product_metrics(df))

print("\nUnique products:")
print(product_metrics["unique_products"])

print("\nTop 10 products by quantity:")
print(product_metrics["quantity_per_product"].head(10))

print("\nTop 10 products by revenue:")
print(product_metrics["revenue_per_product"].head(10))

print("\nTop 10 products by revenue share:")
product_revenue_share = (eda.calculate_product_revenue_share(df))
print(product_revenue_share.head(10))

print("\nCancellation metrics:")
cancellation_metrics = (eda.calculate_cancellation_metrics(df))
print(cancellation_metrics)

print("\nReturn metrics:")
return_metrics = (eda.calculate_return_metrics(df))
print(return_metrics)

print("\nCancellation rate:")
print(eda.calculate_cancellation_rate(df))

