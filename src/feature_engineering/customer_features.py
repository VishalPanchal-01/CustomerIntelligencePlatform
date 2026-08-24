import sys
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logger


class CustomerFeatureEngineering:
    def create_customer_features(self,df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Starting customer feature engineering.")
            data = df.copy()
            data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"],errors="coerce")
            
            data = data.dropna(subset=["InvoiceDate"])
            reference_date = (data["InvoiceDate"].max())
            logger.info(f"Reference date: {reference_date}")

            first_purchase = (data.groupby("Customer ID")["InvoiceDate"].min())
            last_purchase = (data.groupby("Customer ID")["InvoiceDate"].max())

            recency = (reference_date - last_purchase).dt.days
            frequency = (data.groupby("Customer ID")["Invoice"].nunique())
            monetary = (data.groupby("Customer ID")["Revenue"].sum())
            total_items = (data.groupby("Customer ID")["Quantity"].sum())

            customer_features = pd.DataFrame({
                    "Recency": recency,
                    "Frequency": frequency,
                    "Monetary": monetary,
                    "TotalItems": total_items,
                    "FirstPurchaseDate": first_purchase,
                    "LastPurchaseDate": last_purchase
                }
            )

            customer_features.index.name = "Customer ID"
            customer_features = (customer_features.reset_index())
            customer_features["AverageOrderValue"] = (customer_features["Monetary"]/customer_features["Frequency"])

            customer_features["Tenure"] = (customer_features["LastPurchaseDate"]-customer_features["FirstPurchaseDate"]).dt.days

            customer_features = (customer_features.replace([float("inf"), float("-inf")],0))

            customer_features["AverageOrderValue"] = (customer_features["AverageOrderValue"].fillna(0))
            logger.info("Customer feature engineering completed.")
            logger.info(f"Customer feature shape: "f"{customer_features.shape}")

            return customer_features

        except Exception as e:

            logger.error(
                "Customer feature engineering failed."
            )

            raise CustomException(
                e,
                sys
            )