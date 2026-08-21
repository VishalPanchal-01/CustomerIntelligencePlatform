import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logger import logger
from src.utils.exception import CustomException

class EDAVisualization:
    def __init__(self):
        self.figure_dir = os.path.join(os.getcwd(),'artifacts','eda')
        os.makedirs(self.figure_dir , exist_ok=True)

    def plot_monthly_revenue(self,monthly_revenue:pd.Series):
        try:
            logger.info("calculating monthly revenue.")
            plt.figure(figsize=(12,6))

            monthly_revenue.plot(kind='line',marker='o')
            plt.title("Monthly Revenue")
            plt.xlabel('Month')
            plt.ylabel('Revenue')
            plt.xticks(rotation=45)
            plt.tight_layout()

            file_path = os.path.join(self.figure_dir,'monthly_revenue.png')
            plt.savefig(file_path)
            plt.close()
            logger.info(f"Monthly Revenue plot saved : {file_path}")
            return file_path
            
        except Exception as e:
            logger.error("calculating monthly revenue failed.")
            raise CustomException(e,sys) 

    def plot_monthly_orders(self,monthly_orders: pd.Series):
        try:
            logger.info("Creating monthly orders plot.")
            plt.figure(figsize=(12, 6))
            monthly_orders.plot(kind="line",marker="o")
            plt.title("Monthly Orders")
            plt.xlabel("Month")
            plt.ylabel("Number of Orders")
            plt.xticks(rotation=45)
            plt.tight_layout()
            file_path = os.path.join(self.figure_dir,"monthly_orders.png")
            plt.savefig(file_path)
            plt.close()

            logger.info(f"Monthly orders plot saved: "f"{file_path}")
            return file_path

        except Exception as e:
            logger.error("Monthly orders plot failed.")
            raise CustomException(e,sys)       


    def plot_monthly_customers(self,monthly_customers: pd.Series):
        try:
            logger.info("Creating monthly customers plot.")
            plt.figure(figsize=(12, 6))
            monthly_customers.plot(kind="line",marker="o")
            plt.title("Monthly Active Customers")
            plt.xlabel("Month")
            plt.ylabel("Number of Customers")
            plt.xticks(rotation=45)
            plt.tight_layout()
            file_path = os.path.join(
            self.figure_dir,"monthly_customers.png")
            plt.savefig(file_path)
            plt.close()

            logger.info(f"Monthly customers plot saved: "f"{file_path}")
            return file_path

        except Exception as e:
            logger.error("Monthly customers plot failed.")
            raise CustomException(e,sys)
        
    def plot_revenue_distribution(self,df:pd.DataFrame):
        try:
            logger.info("Plotting revenue distribution.")
            plt.figure(figsize=(10,6))
            df['Revenue'].hist(bins=50)
            plt.title('Revenue Distribution')
            plt.xlabel('Revenue')
            plt.ylabel('Frequency')
            plt.tight_layout()

            file_path = os.path.join(self.figure_dir,"revenue_distribution.png")
            plt.savefig(file_path)
            plt.close()

            logger.info(f"Revenue distribution plot saved : {file_path}")
            return file_path

        except Exception as e:
            logger.error("Revenue distribution plot failed.")
            raise CustomException(e,sys)    

    def plot_customer_spending_distribution(self,df:pd.DataFrame):
        try:
            logger.info("plotting customer spending distribution.")
            customer_spending = (df.dropna(subset=['Customer ID']).groupby('Customer ID')['Revenue'].sum())
            plt.figure(figsize=(10,6))
            customer_spending.hist(bins=50)
            plt.title("Customer Spending Distribution")
            plt.xlabel("Total Customer Spending")
            plt.ylabel("Number of customer")
            plt.tight_layout()

            file_path = os.path.join(self.figure_dir,"customer_spending_distribution.png")
            plt.savefig(file_path)
            plt.close()
            logger.info(f"Customer spending distribution plot saved : {file_path}")

            return file_path
        
        except Exception as e:
            logger.error("customer spending failed.")
            raise CustomException(e,sys)    

    def plot_top_countries_by_revenue(self,df:pd.DataFrame,top_n : int =10):
        try:
            logger.info("plotting top countries by revenue.")
            country_revenue = (df.groupby('Country')['Revenue'].sum().sort_values(ascending=False).head(top_n))
            plt.figure(figsize=(10,6))
            country_revenue.sort_values().plot(kind="barh")
            plt.title(f"Top {top_n} countries by revenue")
            plt.xlabel("Country")
            plt.ylabel("Revenue")
            plt.tight_layout()

            file_path = os.path.join(self.figure_dir,"top_countries_by_revenue.png")
            plt.savefig(file_path)
            plt.close()
            logger.info(f"Top countries plot saved : {file_path}")

            return file_path

        except Exception as e:
            logger.error("Top countries plot failed.")
            raise CustomException(e,sys)    

    def plot_top_products_by_revenue(self,df: pd.DataFrame,top_n: int = 10):
        try:
            logger.info("Creating top products by revenue plot.")
            product_revenue = (df.groupby("StockCode")["Revenue"].sum().sort_values(ascending=False).head(top_n))
            plt.figure(figsize=(10, 6))
            product_revenue.sort_values().plot(kind="barh")
            plt.title(f"Top {top_n} Products by Revenue")
            plt.xlabel("Revenue")
            plt.ylabel("Stock Code")
            plt.tight_layout()

            file_path = os.path.join(self.figure_dir,"top_products_by_revenue.png")
            plt.savefig(file_path)
            plt.close()
            logger.info(f"Top products plot saved: "f"{file_path}")

            return file_path

        except Exception as e:
            logger.error("Top products plot failed.")
            raise CustomException(e,sys)

    def plot_top_customers_by_spending(self,df:pd.DataFrame , top_n : int=10):
        try:
            logger.info("Creating plot of top customer by spendings.")

            customer_spending = (df.dropna(subset=['Customer ID']).groupby('Customer ID')['Revenue'].sum().sort_values(ascending=False).head(top_n))
            plt.figure(figsize=(10,6))
            customer_spending.plot(kind="barh")
            plt.title("Top customers by spending")
            plt.xlabel('Total Spending')
            plt.ylabel('Customer ID')
            plt.tight_layout()

            file_path = os.path.join(self.figure_dir,"top_customer_by_spending.png")
            plt.savefig(file_path)
            plt.close()

            logger.info("Top customer by spending plot saved.")
            return file_path

        except Exception as e:
            logger.error("Top customer spending plot failed.")    
            raise CustomException(e,sys)    

    def plot_customer_order_frequency(self,df:pd.DataFrame):
        try:
            logger.info("plotting customer order frequency.")
            customer_order = (df.dropna(subset=['Customer ID']).groupby('Customer ID')['Invoice'].nunique())
            plt.figure(figsize=(10,6))
            customer_order.hist(bins=30)
            plt.title("Customer order frequency Distribution.")
            plt.xlabel("Number of orders")
            plt.ylabel('Number of customers')
            plt.tight_layout()

            file_path = os.path.join(self.figure_dir,"customer_order_frequency.png")
            plt.savefig(file_path)
            plt.close()

            logger.info("customer order frequency plot saved")
            return file_path
        
        except Exception as e:
            logger.error("plotting customer order frequency failed.")
            raise CustomException(e,sys)    