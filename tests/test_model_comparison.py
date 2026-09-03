import pandas as pd

from src.evaluation.model_comparison import ChurnModelComparison

def test_compare_churn_models():
    X_train = pd.DataFrame({
            "Recency": [10,20,30,40,100,120,140,160,180,200],
            "Frequency": [10,9,8,7,5,4,3,2,1,1],
            "Monetary": [1500,1400,1200,1000,700,500,300,200,100,50],
            "TotalItems": [150,140,120,100,70,50,30,20,10,5],
            "AverageOrderValue": [150,155,150,143,140,125,100,100,100,50],
            "Tenure": [350,330,300,280,200,150,100,70,30,10]
        })

    y_train = pd.Series([0,0,0,0,0,1,1,1,1,1])

    X_test = pd.DataFrame({
            "Recency": [15,50,130,190],
            "Frequency": [9,6,2,1],
            "Monetary": [1400,800,250,80],
            "TotalItems": [140,80,25,8],
            "AverageOrderValue": [155,133,125,80],
            "Tenure": [340,240,80,20]
            })


    y_test = pd.Series([0,0,1,1])

    comparator = (ChurnModelComparison())

    result = (comparator.compare_models(X_train,X_test,y_train,y_test))

    assert result is not None

    assert len(result) == 3

    assert "Model" in result.columns

    assert "Accuracy" in result.columns

    assert "Precision" in result.columns

    assert "Recall" in result.columns

    assert "F1Score" in result.columns

    assert "ROCAUC" in result.columns

    assert set(result["Model"]) == {"Baseline","Logistic Regression","Random Forest"}