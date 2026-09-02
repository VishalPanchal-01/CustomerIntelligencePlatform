import sys
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger

from sklearn.model_selection import train_test_split

class ChurnDataSplitter:
    def split_data(self,X:pd.DataFrame,y:pd.Series , test_size : float =0.2 , random_state : int =42):
        try:
            logger.info("Starting churn train test split")
            X_train,X_test , y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=random_state,stratify=y)

            logger.info(f"X_train shape: {X_train.shape}")
            logger.info(f"X_test shape: {X_test.shape}")
            logger.info(f"y_train shape: {y_train.shape}")
            logger.info(f"y_test shape: {y_test.shape}")
            logger.info("Train-test split completed.")

            return (X_train,X_test,y_train,y_test)

        except Exception as e:
            logger.error("")
            raise CustomException(e,sys)
