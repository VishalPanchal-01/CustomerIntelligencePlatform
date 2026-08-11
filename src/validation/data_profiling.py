import os
import sys
import pandas as pd

from src.config.config import ProjectConfig
from src.utils.logger import logger
from src.utils.exception import CustomException

class DataProfiller:
    def __init__(self):
        self.processed_data_path = os.path.join(ProjectConfig.PROCESSED_DATA_DIR,"retail.csv")

    def profile_structure(self):
        try:
            logger.info("Starting Data Structure profilling.")
            if not os.path.exists(self.processed_data_path):
                raise FileNotFoundError(f"Dataset not found at : {self.processed_data_path}")

            df = pd.read_csv(self.processed_data_path)
            rows,columns = df.shape

            logger.info(f"Number of rows : {rows}")
            logger.info(f"Number of coluumns : {columns}")

            logger.info(f"columns Names : {list(df.columns)}")

            logger.info(f"Data Type : \n{df.dtypes}")

            return df 

        except Exception as e:
            logger.error("Data Structure profilling failed.")
            raise CustomException(e,sys)    

    def profile_missing_value(self,df:pd.DataFrame):
        try:
            logger.info("Starting missing value profiling.")
            missing_count = df.isnull().sum()
            missing_percentage = df.isnull().mean() * 100

            missing_report = pd.DataFrame({"missing_count" : missing_count,"missing_percentage" : missing_percentage})
            logger.info(f"Missing value report : \n{missing_report}")

            return missing_report
        
        except Exception as e:
            logger.error("missing value profiling failed.")
            raise CustomException(e,sys)    

    def profile_duplicate(self,df:pd.DataFrame):
        try:
            logger.info("Starting Duplicate profilling.")
            duplicate_count = df.duplicated().sum()
            total_row = len(df)

            duplicate_percentage = (duplicate_count/total_row) * 100

            duplicate_report ={"duplicate_count" : duplicate_count , "duplicate_percentage" : duplicate_percentage}
            logger.info(f"Duplicate report : {duplicate_report}")
            return duplicate_report
        
        except Exception as e:
            logger.error("Duplicate profilling failed.")
            raise CustomException(e,sys)    

    def profile_numeric_columns(self,df:pd.DataFrame):
        try:
            logger.info("Starting Numeric column profiling.")
            quantity_report = {
                "minimum" : df["Quantity"].min(),
                "maximum" : df["Quantity"].max(),
                "mean" : df["Quantity"].mean(),
                "negative_count" : (df["Quantity"]<0).sum(),
                "zero_count" : (df["Quantity"]==0).sum()
            }
            price_report = {
                "minimum" : df["Price"].min(),
                "maximum" : df["Price"].max(),
                "mean" : df["Price"].mean(),
                "negative_count" : (df["Price"]<0).sum(),
                "zero_count" : (df["Price"]==0).sum()
            }
            report = {"quantity" : quantity_report , "price" : price_report}
            logger.info(f"Numerical report : {report}")
            return report

        except Exception as e:
            logger.error("Numeric profiling failed.")
            raise CustomException(e,sys)    


    def profile_cancellation(self,df:pd.DataFrame):
        try:
            logger.info("Starting cancellation profiling.")
            cancellation_mask = df["Invoice"].astype(str).str.startswith('C')
            cancellation_count = cancellation_mask.sum()

            cancellation_percentage = (cancellation_count/len(df)) * 100

            negative_quantity_mask = (df["Quantity"]<0) 
            negative_quantity_count = negative_quantity_mask.sum()
            negative_quantity_percentage = (negative_quantity_count/len(df)) * 100

            report = {
                "cancellation_count" : cancellation_count,
                "cancellation_percentage" : cancellation_percentage,
                "negative_quantity_count" : negative_quantity_count,
                "negative_quantity_percentage" : negative_quantity_percentage
                }
            logger.info(f"cancellation report : {report}")
            return report
        
        except Exception as e:
            logger.error("Cancellation profiling failed.")
            raise CustomException(e,sys)    

    def profile_dates(self, df):
        try:
            logger.info("Starting date profiling")
            invoice_dates = pd.to_datetime(df["InvoiceDate"],errors="coerce")

            invalid_date_count = (invoice_dates.isna().sum())

            date_report = {
                "minimum_date": invoice_dates.min(),
                "maximum_date": invoice_dates.max(),
                "invalid_date_count": invalid_date_count
            }

            logger.info(f"Date report: {date_report}")

            return date_report

        except Exception as e:
            logger.error("Date profiling failed")
            raise CustomException(e,sys)    

    def profile_business_entities(self, df:pd.DataFrame):
        try:
            logger.info("Starting business entity profiling")
            customer_count = (df["Customer ID"].nunique())

            product_count = (df["StockCode"].nunique())
            invoice_count = (df["Invoice"].nunique())

            country_count = (df["Country"].nunique())

            report = {
                "unique_customers": customer_count,
                "unique_products": product_count,
                "unique_invoices": invoice_count,
                "unique_countries": country_count
            }

            logger.info(f"Business entity report: {report}")
            return report

        except Exception as e:
            logger.error("Business entity profiling failed")
            raise CustomException(e,sys)    