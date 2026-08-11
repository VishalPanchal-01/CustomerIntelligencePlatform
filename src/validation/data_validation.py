import os
import sys
import pandas as pd

from src.config.config import ProjectConfig
from src.utils.logger import logger
from src.utils.exception import CustomException

class DataValidation:
    EXPECTED_COLUMNS = [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country"]

    def __init__(self):
        self.processed_data_path = os.path.join(ProjectConfig.PROCESSED_DATA_DIR,"retail.csv")


    def validate_columns(self,df):
        try:
            logger.info("Starting column validation")
            actual_columns = list(df.columns)
            missing_columns = [column for column in self.EXPECTED_COLUMNS if column not in actual_columns]

            if missing_columns:
                raise ValueError(f"Missing required columns : {missing_columns}")
            logger.info("column validation successful.")
            return True
        except Exception as e:
            logger.info("Column validation failed.")
            raise CustomException(e,sys)    

    def validate_data(self):
        try:
            logger.info("Starting the validation")

            if not os.path.exists(self.processed_data_path):
                raise FileNotFoundError(f"processed dataset not found at : {self.processed_data_path}")

            logger.info(f"Reading dataset from : {self.processed_data_path}")
            df = pd.read_csv(self.processed_data_path)

            logger.info(f"Dataset shape : {df.shape}")
            self.validate_columns(df)
            logger.info("Data validation completed successfully.")
            return True
            
        except Exception as e:
            logger.info("Data validaton failed.")
            raise CustomException(e,sys)    




