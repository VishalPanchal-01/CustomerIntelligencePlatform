from src.eda.exploratory_analysis import (
    ExploratoryAnalysis
)
import pandas as pd


def test_eda():

    eda = ExploratoryAnalysis()

    df = eda.load_data()

    assert df is not None
    assert not df.empty

    df = eda.calculate_revenue(df)

    assert "Revenue" in df.columns

    total_revenue = (
        eda.calculate_total_revenue(df)
    )

    assert total_revenue is not None

    order_metrics = (
        eda.calculate_order_metrics(df)
    )

    assert order_metrics is not None

    assert "unique_invoices" in order_metrics

    assert "transaction_rows" in order_metrics

    assert "unique_invoices" in order_metrics

    assert "average_items_per_invoice" in order_metrics

    assert (
        order_metrics["unique_invoices"] > 0
    )

    assert (
        order_metrics["average_items_per_invoice"]
        >= 0
    )

    average_order_value = (
        eda.calculate_average_order_value(df)
    )

    assert average_order_value is not None

    assert average_order_value >= 0


        # Customer metrics

    customer_metrics = (
        eda.calculate_customer_metrics(df)
    )

    assert customer_metrics is not None

    assert (
        "unique_customers"
        in customer_metrics
    )

    assert (
        "average_orders_per_customer"
        in customer_metrics
    )

    assert (
        "average_spending_per_customer"
        in customer_metrics
    )

    assert (
        "average_order_value_per_customer"
        in customer_metrics
    )

    assert (
        customer_metrics[
            "unique_customers"
        ] > 0
    )


        # Date preparation

    df = eda.prepare_date_column(df)

    assert (
        pd.api.types.is_datetime64_any_dtype(
            df["InvoiceDate"]
        )
    )

    # Monthly revenue

    monthly_revenue = (
        eda.calculate_monthly_revenue(df)
    )

    assert monthly_revenue is not None
    assert not monthly_revenue.empty

    # Monthly orders

    monthly_orders = (
        eda.calculate_monthly_orders(df)
    )

    assert monthly_orders is not None
    assert not monthly_orders.empty

    # Monthly customers

    monthly_customers = (
        eda.calculate_monthly_customers(df)
    )

    assert monthly_customers is not None
    assert not monthly_customers.empty


        # Country metrics

    country_metrics = (
        eda.calculate_country_metrics(df)
    )

    assert country_metrics is not None

    assert (
        "customers_per_country"
        in country_metrics
    )

    assert (
        "orders_per_country"
        in country_metrics
    )

    assert (
        "revenue_per_country"
        in country_metrics
    )

    assert (
        not country_metrics[
            "customers_per_country"
        ].empty
    )

    assert (
        not country_metrics[
            "orders_per_country"
        ].empty
    )

    assert (
        not country_metrics[
            "revenue_per_country"
        ].empty
    )


        # Country revenue share

    revenue_share = (eda.calculate_country_revenue_share(df))

    assert revenue_share is not None

    assert not revenue_share.empty



        # Product metrics

    product_metrics = (
        eda.calculate_product_metrics(df)
    )

    assert product_metrics is not None

    assert (
        "unique_products"
        in product_metrics
    )

    assert (
        "quantity_per_product"
        in product_metrics
    )

    assert (
        "revenue_per_product"
        in product_metrics
    )

    assert (
        product_metrics[
            "unique_products"
        ] > 0
    )

    assert (
        not product_metrics[
            "quantity_per_product"
        ].empty
    )

    assert (
        not product_metrics[
            "revenue_per_product"
        ].empty
    )