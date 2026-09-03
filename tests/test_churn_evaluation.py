import pandas as pd

from src.training.baseline_model import BaselineChurnModel

from src.evaluation.churn_evaluation import ChurnModelEvaluation

def test_churn_model_evaluation():
    X_train = pd.DataFrame({
            "Recency": [10,20,30,40]
        })

    y_train = pd.Series([1,1,1,0])

    X_test = pd.DataFrame({
            "Recency": [50,60]
        })

    y_test = pd.Series([1,0])

    trainer = BaselineChurnModel()

    model = trainer.train(X_train,y_train)

    evaluator = (ChurnModelEvaluation())

    result = evaluator.evaluate(model,X_test,y_test)

    assert result is not None

    assert "accuracy" in result

    assert "precision" in result

    assert "recall" in result

    assert "f1_score" in result

    assert "confusion_matrix" in result

    assert "roc_auc" in result