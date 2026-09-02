import sys
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger

class ChurnPreprocessor:
    def prepare_features(self,df:pd.DataFrame):
        try:
            logger.info("Preparing churn features.")

            features = ["Recency","Frequency","Monetary","TotalItems","AverageOrderValue","Tenure"]

            target = "Churn"
            X = df[features].copy()
            logger.info(f"Feature shape: {X.shape}")
            y = df[target].copy()
            logger.info(f"Target shape: {y.shape}")

            if X.isnull().any().any():
                raise ValueError("Feature dataset contains missing values.")

            if y.isnull().any():
                raise ValueError("Target contains missing values.")

            if not set(y.unique()).issubset({0, 1}):
                raise ValueError("Invalid churn labels detected.")

            return X, y

        except Exception as e:
            logger.error("churn prepration fetaure failed.")
            raise CustomException(e,sys)
