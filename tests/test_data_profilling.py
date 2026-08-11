# from src.validation.data_profiling import DataProfiller


# def test_profile_structure():

#     profiler = DataProfiller()

#     df = profiler.profile_structure()

#     assert df is not None
#     assert not df.empty

#     assert len(df.columns) == 8

#     expected_columns = [
#         "Invoice",
#         "StockCode",
#         "Description",
#         "Quantity",
#         "InvoiceDate",
#         "Price",
#         "Customer ID",
#         "Country"
#     ]

#     assert list(df.columns) == expected_columns


from src.validation.data_profiling import DataProfiller


def test_data_profiling():

    profiler = DataProfiller()

    df = profiler.profile_structure()

    missing_report = profiler.profile_missing_value(df)

    assert missing_report is not None

    assert "missing_count" in missing_report.columns

    assert "missing_percentage" in missing_report.columns

    duplicate_report = profiler.profile_duplicate(df)

    assert duplicate_report is not None
    assert "duplicate_count" in duplicate_report
    assert "duplicate_percentage" in duplicate_report

    numeric_report = profiler.profile_numeric_columns(df)

    assert numeric_report is not None
    assert "quantity" in numeric_report
    assert "price" in numeric_report

    cancellation_report = (
        profiler.profile_cancellation(df)
    )

    assert cancellation_report is not None
    assert "cancellation_count" in cancellation_report
    assert "negative_quantity_count" in cancellation_report

    date_report = (
        profiler.profile_dates(df)
    )

    assert date_report is not None
    assert "minimum_date" in date_report
    assert "maximum_date" in date_report

    business_report = (
        profiler.profile_business_entities(df)
    )

    assert business_report is not None
    assert "unique_customers" in business_report
    assert "unique_products" in business_report
    assert "unique_invoices" in business_report
    assert "unique_countries" in business_report