import sys

import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger

from src.training.baseline_model import BaselineChurnModel

from src.training.logistic_model import LogisticChurnModel

from src.training.random_forest_model import RandomForestChurnModel

from src.evaluation.churn_evaluation import ChurnModelEvaluation

class ChurnModelComparison:
    def compare_models(self,X_train: pd.DataFrame,X_test: pd.DataFrame,y_train: pd.Series,y_test: pd.Series) -> pd.DataFrame:
        try:
            logger.info("Starting churn model comparison.")
            evaluator = (ChurnModelEvaluation())
            results = []

            baseline_trainer = (BaselineChurnModel())

            baseline_model = (baseline_trainer.train(X_train,y_train))

            baseline_result = (evaluator.evaluate(baseline_model,X_test,y_test))

            results.append({
                "Model":"Baseline",
                "Accuracy":baseline_result["accuracy"],
                "Precision":baseline_result["precision"],
                "Recall":baseline_result["recall"],
                "F1Score":baseline_result["f1_score"],
                "ROCAUC":baseline_result["roc_auc"]
                })

            logistic_trainer = (LogisticChurnModel())

            logistic_model = (logistic_trainer.train(X_train,y_train))

            logistic_result = (evaluator.evaluate(logistic_model,X_test,y_test))

            results.append({
                "Model":"Logistic Regression",
                "Accuracy":logistic_result["accuracy"],
                "Precision":logistic_result["precision"],
                "Recall":logistic_result["recall"],
                "F1Score":logistic_result["f1_score"],
                "ROCAUC":logistic_result["roc_auc"]
                })

            random_forest_trainer = (RandomForestChurnModel())

            random_forest_model = (random_forest_trainer.train(X_train,y_train))

            random_forest_result = (evaluator.evaluate(random_forest_model,X_test,y_test))

            results.append({
                "Model":"Random Forest",
                "Accuracy":random_forest_result["accuracy"],
                "Precision":random_forest_result["precision"],
                "Recall":random_forest_result["recall"],
                "F1Score":random_forest_result["f1_score"],
                "ROCAUC":random_forest_result["roc_auc"]
                })


            comparison = pd.DataFrame(results)

            comparison = (comparison.sort_values(by="F1Score",ascending=False).reset_index(drop=True))

            logger.info(f"Model comparison:\n"f"{comparison}")

            return comparison

        except Exception as e:
            logger.error("Churn model comparison failed.")
            raise CustomException(e,sys)