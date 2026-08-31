import pandas as pd
from src.analysis.churn_analysis import ChurnAnalysis

def test_analyze_class_distribution():
    df = pd.DataFrame({
            "Churn": [0,0,0,1]
            
        })
    analyzer = ChurnAnalysis()
    result = (analyzer.analyze_class_distribution(df))

    assert result is not None
    assert result.loc[0, "Count"] == 3
    assert result.loc[1, "Count"] == 1
    assert result.loc[0,"Percentage"] == 75.0
    assert result.loc[1,"Percentage"] == 25.0

def test_analyze_feature_by_churn():
    df = pd.DataFrame({
            "Churn": [0,0,1,1],
            "Recency": [10,20,100,120],
            "Frequency": [5,7,1,2],
            "Monetary": [500,700,100,200],
            "TotalItems": [50,70,10,20],
            "AverageOrderValue": [100,100,100,100],
            "Tenure": [200,250,30,40]
        })

    analyzer = ChurnAnalysis()
    result = (analyzer.analyze_feature_by_churn(df))

    assert result is not None
    assert result.loc[0,"Recency"] == 15
    assert result.loc[1,"Recency"] == 110
    assert result.loc[0,"Frequency"] == 6
    assert result.loc[1,"Frequency"] == 1.5