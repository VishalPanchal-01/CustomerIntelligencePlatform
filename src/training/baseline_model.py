import sys
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger

from sklearn.dummy import DummyClassifier

class BaselineChurnModel:
    def train(self,X_train : pd.DataFrame , y_train:pd.Series) ->DummyClassifier:
        try:
            logger.info("Starting baseline churn model training.")

            model = DummyClassifier(strategy="most_frequent")
            model.fit(X_train,y_train)
            logger.info("Baseline Churn model training complete.")

            return model

        except Exception as e:
            logger.error("Bseline churn model training failed.")
            raise CustomException(e,sys)