import pandas as pd

from src.analysis.clv_analysis import (
    CLVAnalysis
)


def create_test_clv_data():

    return pd.DataFrame(
        {
            "Customer ID": [
                101,
                102,
                103,
                104,
                105
            ],

            "Recency": [
                10,
                20,
                50,
                100,
                150
            ],

            "Frequency": [
                10,
                8,
                5,
                2,
                1
            ],

            "Monetary": [
                5000,
                3500,
                1500,
                500,
                100
            ],

            "TotalItems": [
                500,
                350,
                150,
                50,
                10
            ],

            "AverageOrderValue": [
                500,
                437.5,
                300,
                250,
                100
            ],

            "Tenure": [
                400,
                350,
                250,
                100,
                20
            ],

            "FutureRevenue": [
                2000,
                1500,
                500,
                0,
                0
            ]
        }
    )


def test_target_statistics():

    df = create_test_clv_data()

    analyzer = (
        CLVAnalysis()
    )

    result = (
        analyzer
        .analyze_target_statistics(
            df
        )
    )

    assert result is not None

    assert (
        "Mean"
        in result.columns
    )

    assert (
        "Median"
        in result.columns
    )

    assert (
        "Skewness"
        in result.columns
    )

    assert (
        result[
            "Count"
        ].iloc[0]
        == 5
    )


def test_zero_revenue_analysis():

    df = create_test_clv_data()

    analyzer = (
        CLVAnalysis()
    )

    result = (
        analyzer
        .analyze_zero_revenue(
            df
        )
    )

    zero_row = (
        result[
            result[
                "RevenueGroup"
            ]
            ==
            "Zero Future Revenue"
        ]
        .iloc[0]
    )

    assert (
        zero_row[
            "CustomerCount"
        ]
        == 2
    )

    assert (
        zero_row[
            "Percentage"
        ]
        == 40.0
    )


def test_target_quantiles():

    df = create_test_clv_data()

    analyzer = (
        CLVAnalysis()
    )

    result = (
        analyzer
        .analyze_target_quantiles(
            df
        )
    )

    assert result is not None

    assert (
        len(result)
        == 8
    )

    assert (
        "FutureRevenue"
        in result.columns
    )


def test_target_outliers():

    df = create_test_clv_data()

    analyzer = (
        CLVAnalysis()
    )

    result = (
        analyzer
        .analyze_target_outliers(
            df
        )
    )

    assert result is not None

    assert (
        "OutlierCount"
        in result.columns
    )

    assert (
        "UpperBound"
        in result.columns
    )


def test_feature_statistics():

    df = create_test_clv_data()

    analyzer = (
        CLVAnalysis()
    )

    result = (
        analyzer
        .analyze_feature_statistics(
            df
        )
    )

    assert result is not None

    assert (
        len(result)
        == 6
    )

    assert (
        "Feature"
        in result.columns
    )

    assert (
        "mean"
        in result.columns
    )

    assert (
        "skew"
        in result.columns
    )


def test_feature_correlation():

    df = create_test_clv_data()

    analyzer = (
        CLVAnalysis()
    )

    result = (
        analyzer
        .analyze_feature_correlation(
            df
        )
    )

    assert result is not None

    assert (
        len(result)
        == 6
    )

    assert (
        "CorrelationWithFutureRevenue"
        in result.columns
    )


def test_clv_data_quality():

    df = create_test_clv_data()

    analyzer = (
        CLVAnalysis()
    )

    result = (
        analyzer
        .analyze_data_quality(
            df
        )
    )

    assert (
        result[
            "duplicate_customers"
        ]
        == 0
    )

    assert (
        result[
            "infinite_values"
        ]
        == 0
    )

    assert (
        result[
            "negative_future_revenue"
        ]
        == 0
    )