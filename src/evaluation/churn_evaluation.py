import sys
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger

from sklearn.metrics import accuracy_score ,precision_score , recall_score ,confusion_matrix,f1_score,roc_auc_score

class ChurnModelEvaluation:
    def evaluate(self,model,X_test:pd.DataFrame,y_test:pd.Series) -> dict:
        try:
            logger.info("Starting churn model evaluation")
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test,predictions)
            precision = precision_score(y_test,predictions,zero_division=0)
            recall = recall_score(y_test,predictions,zero_division=0)
            f1 = f1_score(y_test,predictions,zero_division=0)
            matrix = confusion_matrix(y_test,predictions)

            if hasattr(model,"predict_proba"):
                probabilities = (model.predict_proba(X_test)[:, 1])

                roc_auc = (roc_auc_score(y_test,probabilities))

            else:   
                roc_auc = None

            report = {
                "accuracy" : accuracy,
                "precision" : precision,
                "recall" : recall,
                "f1_score" : f1,
                "roc_auc":roc_auc,
                "confusion_matrix" : matrix
            }
            logger.info(f"Model evaluation report : {report}")

            return report

        except Exception as e:
            logger.error("Churn model evaluation failed.")
            raise CustomException(e,sys)
