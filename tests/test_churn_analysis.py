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


def test_analyze_feature_statistics():
    df = pd.DataFrame({
            "Recency": [10,20,30],
            "Frequency": [1,2,3],
            "Monetary": [100,200,300],
            "TotalItems": [10,20,30],
            "AverageOrderValue": [100,100,100],
            "Tenure": [50,100,150]
        })

    analyzer = ChurnAnalysis()
    result = (analyzer.analyze_feature_statistics(df))
    assert result is not None
    assert result.loc["Recency","mean"] == 20
    assert result.loc["Recency","median"] == 20
    assert result.loc["Frequency","mean"] == 2
    assert result.loc["Monetary","max"] == 300


def test_analyze_correlation():
    df = pd.DataFrame({
        "Recency": [10,20,30,40],
        "Frequency": [4,3,2,1],
        "Monetary": [400,300,200,100],
        "TotalItems": [40,30,20,10],
        "AverageOrderValue": [100,100,100,100],
        "Tenure": [200,150,100,50],
        "Churn": [0,0,1,1]
        })

    analyzer = ChurnAnalysis()
    result = (analyzer.analyze_correlation(df))

    assert result is not None
    assert (result.shape==(7, 7))
    assert (result.loc["Recency","Recency"]== 1)
    assert (result.loc["Frequency","Monetary"]> 0)    

def test_analyze_feature_quality():
    df = pd.DataFrame({
        "Customer ID": [101,102,103],
        "Recency": [10,20,30],
        "Frequency": [2,3,1],
        "Monetary": [200,300,100],
        "TotalItems": [20,30,10],
        "AverageOrderValue": [100,100,100],
        "Tenure": [50,100,20],
        "Churn": [0,1,1]
        })

    analyzer = ChurnAnalysis()

    result = (analyzer.analyze_churn_quality(df))
    assert result is not None
    assert (result["missing_values"].sum()== 0)
    assert (result["infinite_values"].sum()== 0)
    assert (result["duplicate_customers"]== 0)
    assert (result["invalid_churn_labels"]== 0)
    assert all(value == 0 for value in result["negative_values"].values())    


def test_analyze_feature_skewness():
    df = pd.DataFrame({
            "Recency": [10,20,30,100],
            "Frequency": [1,2,3,20],
            "Monetary": [100,200,300,5000],
            "TotalItems": [10,20,30,500],
            "AverageOrderValue": [100,110,120,1000],
            "Tenure": [10,20,30,100]
        })

    analyzer = ChurnAnalysis()
    result = (analyzer.analyze_feature_skewness(df))

    assert result is not None
    assert "Recency" in result.index
    assert "Frequency" in result.index
    assert "Monetary" in result.index