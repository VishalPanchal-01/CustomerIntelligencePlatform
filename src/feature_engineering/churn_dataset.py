
import sys
import pandas as pd

from src.utils.exception import CustomException 
from src.utils.logger import logger
from src.feature_engineering.customer_features import CustomerFeatureEngineering

class ChurnDatasetBuilder:
    def __init__(self):
        self.feature_engineering = CustomerFeatureEngineering()

    def build_dataset(self,df:pd.DataFrame,prediction_days: int =90) ->pd.DataFrame:
        try:
            logger.info("Start creating churn dataset.")
            data = df.copy()

            data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'],errors='coerce')
            data = data.dropna(subset=['InvoiceDate'])

            dataset_end_date = (data['InvoiceDate'].max())
            observation_cutoff = (dataset_end_date - pd.Timedelta(days=prediction_days))
            logger.info(f"dataset end date : {dataset_end_date}")
            logger.info(f"observation cutoff : {observation_cutoff}")

            observation_data = (data[data['InvoiceDate']<=observation_cutoff].copy())
            logger.info(f"Observation data shape : {observation_data.shape}")
            prediction_data = (data[data['InvoiceDate']>observation_cutoff].copy())
            logger.info(f"Prediction data shape : {prediction_data.shape}")

            observation_customers = (observation_data['Customer ID'].dropna().unique())
            customer_features = self.feature_engineering.create_customer_features(observation_data,observation_cutoff)

            future_customers = set(prediction_data['Customer ID'].dropna().unique())
            customer_features['Churn'] = (~customer_features['Customer ID'].isin(future_customers)).astype(int)

            customer_features = (customer_features[customer_features['Customer ID'].isin(observation_customers)])

            logger.info("Churn dataset created Successfully.")

            logger.info(f"Final churn dataset shape: {customer_features.shape}")
            logger.info(f"Churn distribution:\n {customer_features['Churn'].value_counts()}")

            return customer_features
    
        except Exception as e:
            logger.error("build dataset failed.")
            raise CustomException(e,sys)    