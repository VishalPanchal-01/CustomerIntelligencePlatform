from src.eda.exploratory_analysis import (
    ExploratoryAnalysis
)


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