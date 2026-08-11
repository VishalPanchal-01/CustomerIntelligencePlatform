import os
import sys
import pandas as pd

from src.config.config import ProjectConfig
from src.utils.logger import logger
from src.utils.exception import CustomException

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join(ProjectConfig.RAW_DATA_DIR,"online_retail_II.xlsx")
        self.processed_data_path = os.path.join(ProjectConfig.PROCESSED_DATA_DIR,"retail.csv")


    def initiate_data_ingestion(self):
        try:
            logger.info("Starting Data Ingestion")

            if not os.path.exists(self.raw_data_path):
                raise FileNotFoundError(f"Raw dataset not found at : {self.raw_data_path}")

            os.makedirs(ProjectConfig.PROCESSED_DATA_DIR,exist_ok=True)

            logger.info(f"Reading dataset from : {self.raw_data_path}")

            excel_data = pd.read_excel(self.raw_data_path,sheet_name=None)
            logger.info(f"No. of sheet found : {len(excel_data)}")

            dataframes = []
            for sheet_name , dataframe in excel_data.items():
                logger.info(f"Reading sheet : {sheet_name}")
                dataframes.append(dataframe)

            df = pd.concat(dataframes,ignore_index=True)
            logger.info(f"Combined dataset shape : {df.shape}")    

            df.to_csv(self.processed_data_path,index=False)
            logger.info(f"processed dataset saved into : {self.processed_data_path}")

            return df
        except Exception as e:
            raise CustomException(e,sys)    