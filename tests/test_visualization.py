import os

import pandas as pd

from src.eda.visualization import EDAVisualization


def test_plot_monthly_revenue():

    visualization = EDAVisualization()

    monthly_revenue = pd.Series(
        [1000, 1500, 2000],
        index=pd.to_datetime(
            [
                "2010-01-31",
                "2010-02-28",
                "2010-03-31"
            ]
        )
    )

    file_path = visualization.plot_monthly_revenue(
        monthly_revenue
    )

    assert file_path is not None
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 0


def test_plot_monthly_orders():

    visualization = EDAVisualization()

    monthly_orders = pd.Series(
        [100, 150, 200],
        index=pd.to_datetime(
            [
                "2010-01-31",
                "2010-02-28",
                "2010-03-31"
            ]
        )
    )

    file_path = visualization.plot_monthly_orders(
        monthly_orders
    )

    assert file_path is not None
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 0


def test_plot_monthly_customers():

    visualization = EDAVisualization()

    monthly_customers = pd.Series(
        [50, 75, 100],
        index=pd.to_datetime(
            [
                "2010-01-31",
                "2010-02-28",
                "2010-03-31"
            ]
        )
    )

    file_path = visualization.plot_monthly_customers(
        monthly_customers
    )

    assert file_path is not None
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 0


def test_plot_revenue_distribution():

    visualization = EDAVisualization()

    df = pd.DataFrame(
        {
            "Revenue": [
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100
            ]
        }
    )

    file_path = visualization.plot_revenue_distribution(
        df
    )

    assert file_path is not None
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 0


def test_plot_customer_spending_distribution():

    visualization = EDAVisualization()

    df = pd.DataFrame(
        {
            "Customer ID": [
                101,
                101,
                102,
                102,
                103,
                103
            ],
            "Revenue": [
                100,
                200,
                150,
                250,
                300,
                100
            ]
        }
    )

    file_path = (
        visualization
        .plot_customer_spending_distribution(df)
    )

    assert file_path is not None
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 0


def test_plot_top_countries_by_revenue():

    visualization = EDAVisualization()

    df = pd.DataFrame(
        {
            "Country": [
                "United Kingdom",
                "United Kingdom",
                "Germany",
                "France",
                "Germany"
            ],
            "Revenue": [
                1000,
                500,
                800,
                600,
                200
            ]
        }
    )

    file_path = (
        visualization
        .plot_top_countries_by_revenue(df)
    )

    assert file_path is not None
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 0


def test_plot_top_products_by_revenue():

    visualization = EDAVisualization()

    df = pd.DataFrame(
        {
            "StockCode": [
                "A001",
                "A001",
                "A002",
                "A003",
                "A002"
            ],
            "Revenue": [
                1000,
                500,
                800,
                600,
                200
            ]
        }
    )

    file_path = (
        visualization
        .plot_top_products_by_revenue(df)
    )

    assert file_path is not None
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 0


def test_plot_top_customers_by_spending():

    visualization = EDAVisualization()

    df = pd.DataFrame(
        {
            "Customer ID": [
                101,
                101,
                102,
                102,
                103,
                103,
                104
            ],
            "Revenue": [
                100,
                200,
                150,
                250,
                300,
                100,
                1000
            ]
        }
    )

    file_path = (
        visualization
        .plot_top_customers_by_spending(df)
    )

    assert file_path is not None
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 0


def test_plot_customer_order_frequency():

    visualization = EDAVisualization()

    df = pd.DataFrame(
        {
            "Customer ID": [
                101,
                101,
                102,
                102,
                102,
                103,
                103,
                104
            ],
            "Invoice": [
                "INV001",
                "INV002",
                "INV003",
                "INV004",
                "INV005",
                "INV006",
                "INV007",
                "INV008"
            ]
        }
    )

    file_path = (
        visualization
        .plot_customer_order_frequency(df)
    )

    assert file_path is not None
    assert os.path.exists(file_path)
    assert os.path.getsize(file_path) > 0