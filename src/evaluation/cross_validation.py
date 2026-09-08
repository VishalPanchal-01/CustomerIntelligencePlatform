import sys
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger
from src.training.logistic_model import LogisticChurnModel
from src.training.random_forest_model import RandomForestChurnModel
from sklearn.model_selection import StratifiedKFold , cross_validate


class ChurnCrossValidation:
    def evaluate_models(self,X:pd.DataFrame,y:pd.Series,n_splits :int=5)->pd.DataFrame:
        try:
            logger.info("Starting churn cross-validation.")

            cv = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42)

            models = {"Logistic Regression":LogisticChurnModel().build_model(),"Random Forest":RandomForestChurnModel().build_model()}

            results = []

            for model_name,model in models.items():
                logger.info(f"Cross-validating: {model_name}")

                scores = cross_validate(model,X,y,cv=cv,scoring={"accuracy": "accuracy",
                        "precision": "precision","recall": "recall","f1": "f1","roc_auc": "roc_auc"},n_jobs=-1)

                results.append({
                        "Model":model_name,
                        "AccuracyMean":scores["test_accuracy"].mean(),
                        "AccuracyStd":scores["test_accuracy"].std(),
                        "PrecisionMean":scores["test_precision"].mean(),
                        "RecallMean":scores["test_recall"].mean(),
                        "F1Mean":scores["test_f1"].mean(),
                        "F1Std":scores["test_f1"].std(),
                        "ROCAUCMean":scores["test_roc_auc"].mean(),
                        "ROCAUCStd":scores["test_roc_auc"].std()
                    })

            report = pd.DataFrame(results)

            report = (report.sort_values(by="F1Mean",ascending=False).reset_index(drop=True))

            logger.info(f"Cross-validation results:\n"f"{report}")

            return report

        except Exception as e:
            logger.error("Churn cross-validation failed.")
            raise CustomException(e,sys)