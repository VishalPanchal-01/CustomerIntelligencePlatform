import sys
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class LogisticChurnModel:
    def train(self,X_train:pd.DataFrame,y_train:pd.Series) -> Pipeline:
        try:
            logger.info("Staring logistic regression training")
            model = Pipeline(steps=[("scaler",StandardScaler()) , ("classifier",LogisticRegression(max_iter=1000,random_state=42))])

            model.fit(X_train,y_train)
            logger.info("Logistic regression training completed")

            return model

        except Exception as e:
            logger.error("Logistic regression training failed.")
            raise CustomException(e,sys)