import json
import os

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from src.evaluation.final_churn_report import FinalChurnReport

def test_final_churn_report(tmp_path):
    churn_df = pd.DataFrame({
            "CustomerID": [101,102,103,104],
            "Recency": [10,20,100,120],
            "Frequency": [5,4,2,1],
            "Monetary": [1000,800,200,100],
            "TotalItems": [100,80,20,10],
            "AverageOrderValue": [200,200,100,100],
            "Tenure": [300,250,50,20],
            "Churn": [0,0,1,1]
        })

    X = churn_df[["Recency","Frequency","Monetary","TotalItems","AverageOrderValue","Tenure"]]

    y = churn_df["Churn"]
    model = (RandomForestClassifier(n_estimators=10,random_state=42))
    model.fit(X,y)

    evaluation_results = {
        "accuracy": 0.80,
        "precision": 0.75,
        "recall": 0.90,
        "f1_score": 0.82,
        "roc_auc": 0.88,

        "confusion_matrix":np.array([[8, 2],[1, 9]])
    }

    feature_names = [
        "Recency",
        "Frequency",
        "Monetary",
        "TotalItems",
        "AverageOrderValue",
        "Tenure"
    ]

    output_path = os.path.join(tmp_path,"churn_report.json")

    generator = (FinalChurnReport())

    report = (generator.generate_report(
            model_name="Random Forest",
            model=model,
            evaluation_results=evaluation_results,
            churn_df=churn_df,
            feature_names=feature_names,
            model_path="models/churn/churn_model.pkl",
            prediction_days=90,
            output_path=output_path
        ))

    assert report is not None

    assert os.path.exists(output_path)

    assert (report["model"]["name"]=="Random Forest")

    assert (report["churn_definition"]["prediction_window_days"]== 90)

    assert (report["dataset"]["customers"]== 4)

    assert (report["dataset"]["feature_count"]== 6)

    assert (report["test_metrics"]["accuracy"]== 0.80)
    assert (report["test_metrics"]["confusion_matrix"]==[[8, 2],[1, 9]])

    with open(output_path,"r",encoding="utf-8") as file:
        saved_report = (json.load(file))

    assert (saved_report["model"]["name"]=="Random Forest")