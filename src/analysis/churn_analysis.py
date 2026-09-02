import sys
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger

class ChurnAnalysis:
    def analyze_class_distribution(self,df:pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Analyzing churn class distribution.")
            churn_count = (df['Churn'].value_counts().sort_index())
            churn_percentage = (df["Churn"].value_counts(normalize=True).sort_index() * 100)
            distribution = pd.DataFrame({
                "Count" : churn_count,
                "Percentage" : churn_percentage
            })
            distribution.index.name = "churn"
            logger.info(f"churn distribution : \n {distribution}")

            return distribution

        except Exception as e:
            logger.error("churn class distribution analysis failed.")
            raise CustomException(e,sys)


    def analyze_feature_by_churn(self,df:pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Analyzing customer feature by churn status.")

            features = ['Recency','Frequency','Monetary','TotalItems' , 'AverageOrderValue' ,'Tenure']

            feature_summary = df.groupby('Churn')[features].mean()
            logger.info(f"Feature Summary by Churn : \n {feature_summary}")

            return feature_summary
        
        except Exception as e:
            logger.error("feature by churn analysis failed.")
            raise CustomException(e,sys)    

    def analyze_feature_statistics(self, df:pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Analyzing feature statistics.")
            features = ['Recency','Frequency','Monetary','TotalItems','AverageOrderValue','Tenure']

            statistics = (df[features].agg(['mean','median','std','min','max']).T)
            logger.info(f"Feature Statistics : \n {statistics}")
            return statistics
        
        except Exception as e:
            logger.error("Feature statistics analysis failed.")
            raise CustomException(e,sys)        

    def analyze_correlation(self,df:pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Analyze feature correlations.")
            features = ['Recency','Frequency','Monetary','TotalItems','AverageOrderValue','Tenure','Churn']

            correlation_matrix = (df[features].corr())
            logger.info(f"correlation matrix : /n {correlation_matrix}")
            return correlation_matrix
        
        except Exception as e:
            logger.error("analyzed correlation failed.")
            raise CustomException(e,sys)

    def analyze_churn_quality(self,df:pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Analyzing feature Quality.")
            features = ['Recency','Frequency','Monetary','TotalItems','AverageOrderValue','Tenure']

            missing_values = (df[features + ['Churn']].isnull().sum())
            numeric_data = (df[features].select_dtypes(include='number'))
            infinite_values = (numeric_data.isin([float('-inf'),float('inf')]).sum())
            duplicate_customers = (df['Customer ID'].duplicated().sum())

            negative_values = {}
            for feature in features:
                negative_values[feature] = (df[feature]<0).sum()

            invalid_churn_labels = (~df['Churn'].isin([0,1])).sum()

            report = {
            "missing_values": missing_values,
            "infinite_values": infinite_values,
            "duplicate_customers": duplicate_customers,
            "negative_values": negative_values,
            "invalid_churn_labels": invalid_churn_labels
            }

            logger.info(f"Feature quality report:\n {report}")
            return report    

        except Exception as e:
            logger.error("churn quality analysis failed.")
            raise CustomException(e,sys)

    def analyze_feature_skewness(self,df:pd.DataFrame) ->pd.DataFrame:
        try:
            logger.info("Analyze feature skewness.")
            features = ['Recency','Frequency','Monetary','TotalItems','AverageOrderValue','Tenure']

            skewness = (df[features].skew().sort_values(ascending=False))
            logger.info(f"Feature Skewness : {skewness}")
            return skewness
        
        except Exception as e:
            logger.error("feature skewness analysis failed.")
            raise CustomException(e,sys)    

    def analyze_feature_outliers(self,df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Analyzing feature outliers.")
            features = ["Recency","Frequency","Monetary","TotalItems","AverageOrderValue","Tenure"]

            outlier_report = []
            for feature in features:
                q1 = df[feature].quantile(0.25)
                q3 = df[feature].quantile(0.75)

                iqr = q3 - q1   
                lower_bound = (q1 - 1.5 * iqr)
                upper_bound = (q3 + 1.5 * iqr)

                outliers = df[(df[feature] < lower_bound)|(df[feature] > upper_bound)]
                outlier_count = len(outliers)

                outlier_percentage = (outlier_count/ len(df)* 100)
                outlier_report.append({
                    "Feature": feature,
                    "Q1": q1,
                    "Q3": q3,
                    "IQR": iqr,
                    "LowerBound":lower_bound,
                    "UpperBound":upper_bound,
                    "OutlierCount":outlier_count,
                    "OutlierPercentage":outlier_percentage
                })

                report = pd.DataFrame(outlier_report)

            logger.info(f"Feature outlier report:\n"f"{report}")

            return report

        except Exception as e:
            logger.error("Feature outlier analysis failed.")
            raise CustomException(e,sys)    