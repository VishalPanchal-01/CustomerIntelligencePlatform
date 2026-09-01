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