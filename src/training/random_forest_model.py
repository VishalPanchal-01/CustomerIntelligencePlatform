import sys
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger

from sklearn.ensemble import RandomForestClassifier

class RandomForestChurnModel:
    def train(self,X_train:pd.DataFrame,y_train:pd.Series) -> RandomForestClassifier:
        try:
            logger.info("Start random forest training.")
            model = RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1)
            model.fit(X_train,y_train)
            logger.info("Random forest training complete.")

            return model
        
        except Exception as e:
            logger.error("random forest training failed.")
            raise CustomException(e,sys)