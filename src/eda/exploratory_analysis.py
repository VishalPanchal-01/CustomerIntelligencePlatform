import os
import sys

import pandas as pd

from src.config.config import ProjectConfig
from src.utils.logger import logger
from src.utils.exception import CustomException

class ExploratoryAnalysis:
    def __init__(self):
        self.processed_data_path = os.path.join(ProjectConfig.PROCESSED_DATA_DIR,"retail.csv")

    def load_data(self):
        try:
            logger.info(" Start loading data for EDA.")

            if not os.path.exists(self.processed_data_path):
                raise FileNotFoundError(f"File not found at : {self.processed_data_path}")
            logger.info("Reading Data for EDA.")
            df = pd.read_csv(self.processed_data_path)
            logger.info(f"EDA dataset shape : {df.shape}")
            return df
        
        except Exception as e:
            logger.error("Failed to load data for EDA.")
            raise CustomException(e,sys)  

    def calculate_revenue(self,df:pd.DataFrame):
        try:
            logger.info("Calculating transaction revenue.")
            df = df.copy()
            df["Revenue"] = df["Quantity"] * df["Price"]
            logger.info("Revenue column created successfuly.")
            return df
        
        except Exception as e:
            logger.error("Revenue Calculation Falied.")
            raise CustomException(e,sys)      

    def calculate_total_revenue(self,df:pd.DataFrame):
        try:
            logger.info("Calculating total revenue.")
            total_revenue = df["Revenue"].sum()
            logger.info(f"Total Revenue : {total_revenue}")
            return total_revenue
        
        except Exception as e:
            logger.error("Total revenue calculation failed.")    


    def calculate_order_metrics(self,df:pd.DataFrame):
        try:
            logger.info("Calculating order metrics.")
            unique_invoices = df["Invoice"].nunique() 
            transaction_rows = len(df)  
            items_per_invoice = ( 
            df.groupby("Invoice")["Quantity"] .sum() ) 
 
            average_items_per_invoice = items_per_invoice.mean() 
 
            report = { 
            "unique_invoices": unique_invoices, 
            "transaction_rows": transaction_rows, 
            "average_items_per_invoice": average_items_per_invoice } 
 
            return report
        
        except Exception as e:
            logger.error("order metris calculation failed.")
            raise CustomException(e,sys)      

    def calculate_average_order_value(self,df: pd.DataFrame):
        try:
            logger.info("Calculating average order value.")
            total_revenue = (df["Revenue"].sum())
            unique_invoices = (df["Invoice"].nunique())

            if unique_invoices == 0:
                return 0

            average_order_value = (total_revenue /unique_invoices)
            logger.info(f"Average order value: "f"{average_order_value}")

            return average_order_value

        except Exception as e:
            logger.error("Average order value calculation failed.")
            raise CustomException(e, sys)

    def calculate_customer_metrics(self,df:pd.DataFrame):
        try:
            logger.info(" Calculating Customer metrics.")
            df_customer = df.dropna(subset=['Customer ID']).copy()

            unique_customers = df_customer['Customer ID'].nunique()
            orders_per_customer = (df_customer.groupby('Customer ID')['Invoice'].nunique())
            spending_per_customer = (df_customer.groupby('Customer ID')['Revenue'].sum())
            average_order_value_per_customer = (spending_per_customer/orders_per_customer)

            report = {"unique_customers" : unique_customers , 'average_orders_per_customer' : orders_per_customer.mean(),'average_spending_per_customer':spending_per_customer.mean(),'average_order_value_per_customer' : average_order_value_per_customer.mean()}

            logger.info(f"metrics report : {report}")
            return report
        
        except Exception as e:
            logger.error("Customer metrics failed.")
            raise CustomException(e,sys)
