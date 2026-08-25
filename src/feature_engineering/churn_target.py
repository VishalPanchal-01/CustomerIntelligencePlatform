import os 
import sys
import pandas as pd

from src.config.config import ProjectConfig
from src.utils.exception import CustomException
from src.utils.logger import logger

class ChurnTargetCreation:
    def create_churn_target(self,df:pd.DataFrame,prediction_days : int=90)->pd.DataFrame:
        try:
            logger.info("craeting churn target.")
            data = df.copy()
            data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'],errors='coerce')
            data = data.dropna(subset=['InvoiceDate'])

            dataset_end_date = (data['InvoiceDate'].max())
            observation_cutoff = (dataset_end_date - pd.Timedelta(days=prediction_days))

            logger.info(f"dataset_end_date : {dataset_end_date}")
            logger.info(f"observation cutoff : {observation_cutoff}")
            observation_data = (data[data['InvoiceDate']<=observation_cutoff].copy())
            prediction_data = (data[data['InvoiceDate']>observation_cutoff].copy())
            observation_customers = set(observation_data['Customer ID'].dropna().unique())
            future_customers  = set(prediction_data['Customer ID'].dropna().unique())

            customer_target = pd.DataFrame({
                "Customer ID" :list(observation_customers)
            }) 
            customer_target["Churn"] = (customer_target["Customer ID"].apply(lambda customer_id:0 if customer_id in future_customers else 1))

            logger.info("churn target created successfully.")
            logger.info(f"Target shape : {customer_target.shape}")
            logger.info(f"churn distribution : \n {customer_target['Churn'].value_counts()}")

            return customer_target
        
        except Exception as e:
            logger.error("churn target creation failed.")
            raise CustomException(e,sys)