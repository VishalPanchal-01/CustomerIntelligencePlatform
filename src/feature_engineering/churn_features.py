import os
import sys

import pandas as pd

from src.config.config import ProjectConfig
from src.utils.exception import CustomException
from src.utils.logger import logger

class ChurnFeatureEngineer:
    def __init__(self):
        self.output_dir = os.path.join(os.getcwd(),'artifacts','churn')
        os.makedirs(self.output_dir,exist_ok=True)

    def create_customer_features(self,df:pd.DataFrame,observation_date:pd.Timestamp) ->pd.DataFrame:
        try:
            logger.info("Creating customer feature.")
            df = df.copy()
            df = df.dropna(subset=['Customer ID'])
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']) 

            # Recency:
            last_purchase = (df.groupby('Customer ID')['InvoiceDate'].max())
            recency = (observation_date - last_purchase).dt.days

            # Frequency:
            frequency = (df.groupby('Customer ID')['Invoice'].nunique())

            # Monetary:
            monetary = (df.groupby('Customer ID')['Revenue'].sum())

            # Average Order Value:
            average_order_value = (monetary/frequency)

            # First Purchase:
            first_purchase = (df.groupby('Customer ID')['InvoiceDate'].min())

            # Customer Tenure:
            tenure = (observation_date - first_purchase).dt.days

            customer_features = pd.DataFrame({
                "Recency": recency,
                "Frequency" : frequency,
                "Monetary" : monetary,
                "AverageOrderValue" :average_order_value,
                "Tenure" : tenure
            })
            customer_features.index.name = "Customer ID"
            customer_features = (customer_features.reset_index())

            logger.info("Customer level churn feature created successfully.")
            return customer_features
        
        except Exception as e:
            logger.error("customer feature creation failed.")
            raise CustomException(e,sys)    


    def create_churn_label(self,df:pd.DataFrame,customer_features:pd.DataFrame,observation_date:pd.Timestamp,churn_window_days:int = 90) ->pd.DataFrame:
        try:
            logger.info("Creating churn label.")
            df = df.copy()
            df = df.dropna(subset=['Customer ID'])
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

            prediction_end_date = (observation_date + pd.Timedelta(days=churn_window_days))

            future_transactions = df[(df["InvoiceDate"] > observation_date)&(df["InvoiceDate"] <= prediction_end_date)]

            future_customers = set(future_transactions["Customer ID"].unique())
            customer_features["Churn"] = (~customer_features["Customer ID"].isin(future_customers)).astype(int)

            logger.info("Churn labels created successfully.")
            return customer_features
        
        except Exception as e:
            raise CustomException(e,sys)    


    def build_churn_dataset(self,df: pd.DataFrame,observation_date: pd.Timestamp,churn_window_days: int = 90) -> pd.DataFrame:
        try:
            logger.info("Building complete churn dataset.")
            customer_features = (self.create_customer_features(df,observation_date))

            churn_dataset = (self.create_churn_label(df,customer_features,observation_date,churn_window_days))

            output_path = os.path.join(self.output_dir,"customer_churn_dataset.csv")
            churn_dataset.to_csv(output_path,index=False)
            logger.info(f"Churn dataset saved: {output_path}")

            return churn_dataset

        except Exception as e:
            logger.error("churn dataset creation failed.")
            raise CustomException(e,sys)    