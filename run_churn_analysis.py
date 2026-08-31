import pandas as pd

from src.analysis.churn_analysis import ChurnAnalysis

def main():
    df = pd.read_csv("artifacts/churn/customer_churn_dataset.csv")
    analyzer = ChurnAnalysis()
    distribution = (analyzer.analyze_class_distribution(df))
    print("\nChurn Class Distribution:")
    print(distribution)


    feature_summary = (analyzer.analyze_feature_by_churn(df))
    print("\nFeature Summary by Churn:")
    print(feature_summary)    


if __name__ == "__main__":
    main()