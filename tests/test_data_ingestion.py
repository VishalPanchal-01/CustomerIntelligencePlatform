import os

from src.ingestion.data_ingestion import DataIngestion


def test_data_ingestion():

    data_ingestion = DataIngestion()

    df = data_ingestion.initiate_data_ingestion()

    assert df is not None
    assert not df.empty

    assert os.path.exists(
        data_ingestion.processed_data_path
    )