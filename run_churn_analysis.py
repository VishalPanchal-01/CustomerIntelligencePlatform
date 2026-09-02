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

    statistics = (analyzer.analyze_feature_statistics(df))
    print("\nFeature Statistics:")
    print(statistics)


    correlation = (analyzer.analyze_correlation(df))

    print("\nCorrelation Matrix:")
    print(correlation)


    quality_report = (analyzer.analyze_churn_quality(df))

    print("\nFeature Quality Report:")

    print("\nMissing Values:")

    print(quality_report["missing_values"])

    print("\nInfinite Values:")

    print(quality_report["infinite_values"])
    print("\nDuplicate Customers:")
    print(quality_report["duplicate_customers"])

    print("\nNegative Values:")

    print(quality_report["negative_values"])

    print("\nInvalid Churn Labels:")
    print(quality_report["invalid_churn_labels"])



    skewness = (analyzer.analyze_feature_skewness(df))

    print("\nFeature Skewness:")

    print(skewness)

    outliers = (analyzer.analyze_feature_outliers(df))

    print("\nFeature Outlier Report:")
    print(outliers)

if __name__ == "__main__":
    main()