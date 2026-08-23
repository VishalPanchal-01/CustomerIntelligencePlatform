import sys 
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger

class DataCleaning:
    def clean_for_churn(self, df:pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("cleaning churn dataset.")
            cleaned_df = df.copy()
            cleaned_df['InvoiceDate'] = pd.to_datetime(cleaned_df['InvoiceDate'],errors='coerce')
            cleaned_df = cleaned_df.dropna(subset=['InvoiceDate'])
            cleaned_df = cleaned_df.dropna(subset=['Customer ID'])

            cleaned_df = cleaned_df[(cleaned_df['Quantity']>0) & (cleaned_df['Price']>0)]
            cleaned_df = cleaned_df[~cleaned_df['Invoice'].astype(str).str.startswith('C')]
            cleaned_df['Revenue'] = (cleaned_df['Quantity'] * cleaned_df['Price'])

            logger.info(f"cleaned churn dataset shape : {cleaned_df.shape}")
            return cleaned_df
        
        except Exception as e:
            logger.error("cleaning churn dataset failed.")
            raise CustomException(e,sys)